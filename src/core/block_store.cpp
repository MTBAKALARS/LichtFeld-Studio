/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/block_store.hpp"

#include "core/logger.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfs::core {

    namespace {

        // Filenames inside the store directory.
        constexpr const char* kBaseFile = "base.bin";
        constexpr const char* kBoundsFile = "bounds.bin";
        constexpr const char* kIndexFile = "index.bin";
        constexpr const char* kManifestFile = "manifest.bin";

        // Magic + version for the manifest header (forward compatibility).
        constexpr std::uint32_t kManifestMagic = 0x4C544253u; // 'LTBS' (LichtFeld-Tide Block Store)
        constexpr std::uint32_t kManifestVersion = 1;

        struct Manifest {
            std::uint32_t magic;
            std::uint32_t version;
            std::uint64_t num_blocks;
            std::uint64_t block_size;       // Gaussians per block
            std::uint64_t bytes_per_block;  // = block_size * 236
            std::uint64_t patch_segment_capacity_bytes;
        };
        static_assert(sizeof(Manifest) == 40, "Manifest layout must be stable");

        std::string filesystem_error_to_string(const std::filesystem::filesystem_error& e) {
            return std::string{"filesystem error: "} + e.what();
        }

#if defined(_WIN32)
        std::string last_win32_error(const char* prefix) {
            DWORD err = GetLastError();
            char* buf = nullptr;
            FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                           reinterpret_cast<LPSTR>(&buf), 0, nullptr);
            std::string msg = std::string{prefix} + ": [" + std::to_string(err) + "] " + (buf ? buf : "");
            if (buf) LocalFree(buf);
            return msg;
        }
#endif

    } // namespace

    /**
     * @brief Pimpl holding the OS-specific mmap state and segment table.
     */
    struct BlockStore::Impl {
        // Base segment (immutable, mmapped read-only after creation).
        const std::byte* base_view = nullptr;
        std::size_t base_view_size = 0;

#if defined(_WIN32)
        HANDLE base_file = INVALID_HANDLE_VALUE;
        HANDLE base_mapping = nullptr;
        // Patch segments: append-only files, opened for write + memory-mapped read view (per segment).
        struct PatchSegment {
            HANDLE file = INVALID_HANDLE_VALUE;
            HANDLE mapping = nullptr;
            const std::byte* view = nullptr;
            std::uint64_t view_size = 0;
            std::uint64_t write_offset = 0;
        };
#else
        int base_fd = -1;
        struct PatchSegment {
            int fd = -1;
            const std::byte* view = nullptr;
            std::uint64_t view_size = 0;
            std::uint64_t write_offset = 0;
        };
#endif
        std::vector<PatchSegment> patch_segments;
        std::filesystem::path dir;

        // Index: latest-version pointer per block. Guarded by index_mutex.
        std::vector<IndexEntry> index;
        mutable std::mutex index_mutex;

        // Patch segment append guard (one writer at a time per active patch).
        std::mutex patch_mutex;

        // Append a block payload to the active patch segment, rolling over if needed.
        // Returns the resolved IndexEntry pointing at the new copy.
        std::expected<IndexEntry, std::string> append_to_patch(std::span<const std::byte> src,
                                                               std::uint32_t new_version,
                                                               std::uint64_t patch_segment_capacity);

        // Open a patch segment file for append (creates if missing).
        std::expected<PatchSegment, std::string> open_patch_segment(std::uint32_t file_id);

        // Persist the current index to disk (atomic via temp file + rename).
        std::expected<void, std::string> persist_index();

        // Map the base segment read-only.
        std::expected<void, std::string> map_base(const std::filesystem::path& path);

        // Cleanup all OS handles.
        void unmap_all();

        ~Impl() { unmap_all(); }
    };

    // ============================================================
    // Construction / move / destruction
    // ============================================================

    BlockStore::BlockStore() = default;
    BlockStore::~BlockStore() {
        // close() is idempotent; ignore error path here (logged inside).
        (void)close();
    }
    // Move operations are deleted in the header (mutex member).

    // ============================================================
    // create / open
    // ============================================================

    std::expected<std::unique_ptr<BlockStore>, std::string> BlockStore::create(
        const std::filesystem::path& dir,
        std::size_t num_blocks,
        std::span<const BlockBounds> block_bounds,
        std::span<const std::byte> base_bytes,
        const Config& config) {

        if (block_bounds.size() != num_blocks) {
            return std::unexpected{std::format("create: block_bounds.size()={} != num_blocks={}",
                                               block_bounds.size(), num_blocks)};
        }
        const std::size_t bytes_per_block = config.block_size * kBytesPerGaussian;
        const std::size_t expected_bytes = num_blocks * bytes_per_block;
        if (base_bytes.size() != expected_bytes) {
            return std::unexpected{std::format("create: base_bytes.size()={} != expected={} ({} blocks * {} B)",
                                               base_bytes.size(), expected_bytes, num_blocks, bytes_per_block)};
        }

        try {
            std::filesystem::create_directories(dir);
        } catch (const std::filesystem::filesystem_error& e) {
            return std::unexpected{filesystem_error_to_string(e)};
        }

        // Refuse to clobber an existing store.
        if (std::filesystem::exists(dir / kManifestFile)) {
            return std::unexpected{std::format("create: store already exists at {}", dir.string())};
        }

        // Write base.bin sequentially (mmap-on-open is fine for read-only base).
        {
            std::ofstream out(dir / kBaseFile, std::ios::binary);
            if (!out) return std::unexpected{"create: failed to open base.bin for write"};
            out.write(reinterpret_cast<const char*>(base_bytes.data()),
                      static_cast<std::streamsize>(base_bytes.size()));
            if (!out) return std::unexpected{"create: write base.bin failed"};
        }

        // Write bounds.bin
        {
            std::ofstream out(dir / kBoundsFile, std::ios::binary);
            if (!out) return std::unexpected{"create: failed to open bounds.bin for write"};
            out.write(reinterpret_cast<const char*>(block_bounds.data()),
                      static_cast<std::streamsize>(block_bounds.size_bytes()));
            if (!out) return std::unexpected{"create: write bounds.bin failed"};
        }

        // Write initial index.bin: every block points at base segment.
        {
            std::vector<IndexEntry> entries(num_blocks);
            for (std::size_t k = 0; k < num_blocks; ++k) {
                entries[k] = IndexEntry{
                    .file_id = 0, // base
                    .offset = static_cast<std::uint64_t>(k) * bytes_per_block,
                    .size = static_cast<std::uint32_t>(bytes_per_block),
                    .version = 0,
                };
            }
            std::ofstream out(dir / kIndexFile, std::ios::binary);
            if (!out) return std::unexpected{"create: failed to open index.bin for write"};
            out.write(reinterpret_cast<const char*>(entries.data()),
                      static_cast<std::streamsize>(entries.size() * sizeof(IndexEntry)));
            if (!out) return std::unexpected{"create: write index.bin failed"};
        }

        // Write manifest last — its presence signals a fully-initialized store.
        {
            Manifest m{
                .magic = kManifestMagic,
                .version = kManifestVersion,
                .num_blocks = num_blocks,
                .block_size = config.block_size,
                .bytes_per_block = bytes_per_block,
                .patch_segment_capacity_bytes = config.patch_segment_capacity_bytes,
            };
            std::ofstream out(dir / kManifestFile, std::ios::binary);
            if (!out) return std::unexpected{"create: failed to open manifest.bin for write"};
            out.write(reinterpret_cast<const char*>(&m), sizeof(m));
            if (!out) return std::unexpected{"create: write manifest.bin failed"};
        }

        LOG_INFO("BlockStore: created at {} ({} blocks, {} B/block, {:.2f} GiB)",
                 dir.string(), num_blocks, bytes_per_block,
                 static_cast<double>(expected_bytes) / (1ull << 30));

        return open(dir, config);
    }

    std::expected<std::unique_ptr<BlockStore>, std::string> BlockStore::open(
        const std::filesystem::path& dir,
        const Config& config_in) {

        const auto manifest_path = dir / kManifestFile;
        if (!std::filesystem::exists(manifest_path)) {
            return std::unexpected{std::format("open: no manifest at {}", manifest_path.string())};
        }

        // Read manifest first — it determines block_size, num_blocks, etc.
        Manifest m{};
        {
            std::ifstream in(manifest_path, std::ios::binary);
            if (!in) return std::unexpected{"open: failed to open manifest.bin"};
            in.read(reinterpret_cast<char*>(&m), sizeof(m));
            if (!in || m.magic != kManifestMagic) {
                return std::unexpected{"open: bad manifest magic"};
            }
            if (m.version != kManifestVersion) {
                return std::unexpected{std::format("open: unsupported manifest version {}", m.version)};
            }
        }

        auto store = std::unique_ptr<BlockStore>(new BlockStore{});
        store->dir_ = dir;
        store->config_ = config_in;
        store->config_.block_size = m.block_size;
        store->config_.patch_segment_capacity_bytes = m.patch_segment_capacity_bytes;
        store->num_blocks_ = m.num_blocks;
        store->stats_ = std::make_unique<Stats>();
        store->impl_ = std::make_unique<Impl>();
        store->impl_->dir = dir;

        // Load bounds
        store->bounds_.resize(m.num_blocks);
        {
            std::ifstream in(dir / kBoundsFile, std::ios::binary);
            if (!in) return std::unexpected{"open: failed to open bounds.bin"};
            in.read(reinterpret_cast<char*>(store->bounds_.data()),
                    static_cast<std::streamsize>(store->bounds_.size() * sizeof(BlockBounds)));
            if (!in) return std::unexpected{"open: read bounds.bin failed"};
        }

        // Load index
        store->impl_->index.resize(m.num_blocks);
        {
            std::ifstream in(dir / kIndexFile, std::ios::binary);
            if (!in) return std::unexpected{"open: failed to open index.bin"};
            in.read(reinterpret_cast<char*>(store->impl_->index.data()),
                    static_cast<std::streamsize>(store->impl_->index.size() * sizeof(IndexEntry)));
            if (!in) return std::unexpected{"open: read index.bin failed"};
        }

        // mmap base segment
        if (auto r = store->impl_->map_base(dir / kBaseFile); !r) {
            return std::unexpected{r.error()};
        }

        // Discover and map any pre-existing patch segments (file ids 1..N)
        for (std::uint32_t fid = 1; ; ++fid) {
            const auto path = dir / std::format("patch_{:06}.bin", fid);
            if (!std::filesystem::exists(path)) break;
            auto seg = store->impl_->open_patch_segment(fid);
            if (!seg) return std::unexpected{seg.error()};
            store->impl_->patch_segments.push_back(std::move(*seg));
            store->stats_->patch_segments.fetch_add(1, std::memory_order_relaxed);
        }

        LOG_INFO("BlockStore: opened {} ({} blocks, {} patch segments)",
                 dir.string(), m.num_blocks, store->impl_->patch_segments.size());

        return store;
    }

    std::expected<void, std::string> BlockStore::close() {
        if (!impl_) return {};
        // Persist the index one last time so any in-flight patches are recoverable.
        if (auto r = impl_->persist_index(); !r) {
            LOG_ERROR("BlockStore::close: persist_index failed: {}", r.error());
        }
        impl_->unmap_all();
        impl_.reset();
        return {};
    }

    // ============================================================
    // read_block / write_block
    // ============================================================

    std::expected<void, std::string> BlockStore::read_block(std::size_t block_id,
                                                            std::span<std::byte> dst) const {
        if (!impl_) return std::unexpected{"read_block: store closed"};
        if (block_id >= num_blocks_) {
            return std::unexpected{std::format("read_block: block_id={} >= num_blocks={}", block_id, num_blocks_)};
        }
        if (dst.size() != bytes_per_block()) {
            return std::unexpected{std::format("read_block: dst.size()={} != bytes_per_block={}",
                                               dst.size(), bytes_per_block())};
        }

        IndexEntry entry;
        {
            std::scoped_lock lk(impl_->index_mutex);
            entry = impl_->index[block_id];
        }

        const std::byte* src = nullptr;
        std::size_t src_avail = 0;
        if (entry.file_id == 0) {
            src = impl_->base_view + entry.offset;
            src_avail = impl_->base_view_size - entry.offset;
        } else {
            const std::uint32_t idx = entry.file_id - 1;
            if (idx >= impl_->patch_segments.size()) {
                return std::unexpected{std::format("read_block: stale file_id={} for block {}",
                                                   entry.file_id, block_id)};
            }
            const auto& seg = impl_->patch_segments[idx];
            src = seg.view + entry.offset;
            src_avail = seg.view_size - entry.offset;
        }
        if (src_avail < entry.size) {
            return std::unexpected{"read_block: index points past mapped view"};
        }
        std::memcpy(dst.data(), src, entry.size);
        stats_->reads.fetch_add(1, std::memory_order_relaxed);
        stats_->bytes_read.fetch_add(entry.size, std::memory_order_relaxed);
        return {};
    }

    std::expected<std::uint32_t, std::string> BlockStore::write_block(std::size_t block_id,
                                                                     std::span<const std::byte> src) {
        if (!impl_) return std::unexpected{"write_block: store closed"};
        if (block_id >= num_blocks_) {
            return std::unexpected{std::format("write_block: block_id={} >= num_blocks={}", block_id, num_blocks_)};
        }
        if (src.size() != bytes_per_block()) {
            return std::unexpected{std::format("write_block: src.size()={} != bytes_per_block={}",
                                               src.size(), bytes_per_block())};
        }

        // Read previous version to compute new version.
        std::uint32_t new_version;
        {
            std::scoped_lock lk(impl_->index_mutex);
            new_version = impl_->index[block_id].version + 1;
        }

        auto appended = impl_->append_to_patch(src, new_version, config_.patch_segment_capacity_bytes);
        if (!appended) return std::unexpected{appended.error()};

        // Atomically update the index entry.
        {
            std::scoped_lock lk(impl_->index_mutex);
            impl_->index[block_id] = *appended;
        }

        stats_->writes.fetch_add(1, std::memory_order_relaxed);
        stats_->bytes_written.fetch_add(src.size(), std::memory_order_relaxed);
        return new_version;
    }

    std::expected<BlockStore::IndexEntry, std::string> BlockStore::lookup(std::size_t block_id) const {
        if (!impl_) return std::unexpected{"lookup: store closed"};
        if (block_id >= num_blocks_) {
            return std::unexpected{std::format("lookup: block_id={} >= num_blocks={}", block_id, num_blocks_)};
        }
        std::scoped_lock lk(impl_->index_mutex);
        return impl_->index[block_id];
    }

    BlockStore::BlockBounds BlockStore::get_bounds(std::size_t block_id) const {
        std::scoped_lock lk(bounds_mutex_);
        return bounds_[block_id];
    }

    void BlockStore::snapshot_bounds(std::vector<BlockBounds>& out) const {
        std::scoped_lock lk(bounds_mutex_);
        out.assign(bounds_.begin(), bounds_.end());
    }

    void BlockStore::update_bounds(std::size_t block_id, const BlockBounds& bounds) {
        std::scoped_lock lk(bounds_mutex_);
        bounds_[block_id] = bounds;
    }

    // ============================================================
    // Impl: OS-specific mmap and patch append
    // ============================================================

    std::expected<void, std::string> BlockStore::Impl::map_base(const std::filesystem::path& path) {
#if defined(_WIN32)
        base_file = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (base_file == INVALID_HANDLE_VALUE) return std::unexpected{last_win32_error("CreateFile(base)")};

        LARGE_INTEGER size;
        if (!GetFileSizeEx(base_file, &size)) return std::unexpected{last_win32_error("GetFileSizeEx(base)")};
        base_view_size = static_cast<std::size_t>(size.QuadPart);

        base_mapping = CreateFileMappingW(base_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!base_mapping) return std::unexpected{last_win32_error("CreateFileMapping(base)")};

        void* view = MapViewOfFile(base_mapping, FILE_MAP_READ, 0, 0, 0);
        if (!view) return std::unexpected{last_win32_error("MapViewOfFile(base)")};
        base_view = static_cast<const std::byte*>(view);
        return {};
#else
        base_fd = ::open(path.c_str(), O_RDONLY);
        if (base_fd < 0) return std::unexpected{std::string{"open(base) failed: "} + std::strerror(errno)};
        struct stat st{};
        if (::fstat(base_fd, &st) != 0) return std::unexpected{"fstat(base) failed"};
        base_view_size = static_cast<std::size_t>(st.st_size);
        void* view = ::mmap(nullptr, base_view_size, PROT_READ, MAP_SHARED, base_fd, 0);
        if (view == MAP_FAILED) return std::unexpected{"mmap(base) failed"};
        base_view = static_cast<const std::byte*>(view);
        return {};
#endif
    }

    std::expected<BlockStore::Impl::PatchSegment, std::string>
    BlockStore::Impl::open_patch_segment(std::uint32_t file_id) {
        const auto path = dir / std::format("patch_{:06}.bin", file_id);
        PatchSegment seg{};
#if defined(_WIN32)
        seg.file = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (seg.file == INVALID_HANDLE_VALUE) return std::unexpected{last_win32_error("CreateFile(patch)")};
        LARGE_INTEGER size;
        if (!GetFileSizeEx(seg.file, &size)) return std::unexpected{last_win32_error("GetFileSizeEx(patch)")};
        seg.write_offset = static_cast<std::uint64_t>(size.QuadPart);
        seg.view_size = seg.write_offset; // Read view extends to current EOF; we remap on growth.
        if (seg.view_size > 0) {
            seg.mapping = CreateFileMappingW(seg.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (!seg.mapping) return std::unexpected{last_win32_error("CreateFileMapping(patch)")};
            void* view = MapViewOfFile(seg.mapping, FILE_MAP_READ, 0, 0, 0);
            if (!view) return std::unexpected{last_win32_error("MapViewOfFile(patch)")};
            seg.view = static_cast<const std::byte*>(view);
        }
        return seg;
#else
        seg.fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
        if (seg.fd < 0) return std::unexpected{std::string{"open(patch) failed: "} + std::strerror(errno)};
        struct stat st{};
        if (::fstat(seg.fd, &st) != 0) return std::unexpected{"fstat(patch) failed"};
        seg.write_offset = static_cast<std::uint64_t>(st.st_size);
        seg.view_size = seg.write_offset;
        if (seg.view_size > 0) {
            void* view = ::mmap(nullptr, seg.view_size, PROT_READ, MAP_SHARED, seg.fd, 0);
            if (view == MAP_FAILED) return std::unexpected{"mmap(patch) failed"};
            seg.view = static_cast<const std::byte*>(view);
        }
        return seg;
#endif
    }

    std::expected<BlockStore::IndexEntry, std::string>
    BlockStore::Impl::append_to_patch(std::span<const std::byte> src,
                                      std::uint32_t new_version,
                                      std::uint64_t patch_segment_capacity) {
        std::scoped_lock lk(patch_mutex);

        // Roll over to a new patch segment if the active one is full or none yet.
        const bool need_new_segment =
            patch_segments.empty() ||
            patch_segments.back().write_offset + src.size() > patch_segment_capacity;

        if (need_new_segment) {
            const std::uint32_t next_id = static_cast<std::uint32_t>(patch_segments.size()) + 1;
            auto seg = open_patch_segment(next_id);
            if (!seg) return std::unexpected{seg.error()};
            patch_segments.push_back(std::move(*seg));
        }

        auto& active = patch_segments.back();
        const std::uint64_t offset = active.write_offset;

#if defined(_WIN32)
        // Write at end. We opened with FILE_APPEND-equivalent semantics by tracking write_offset ourselves.
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(active.file, li, nullptr, FILE_BEGIN)) {
            return std::unexpected{last_win32_error("SetFilePointerEx(patch append)")};
        }
        DWORD written = 0;
        if (!WriteFile(active.file, src.data(), static_cast<DWORD>(src.size()), &written, nullptr)) {
            return std::unexpected{last_win32_error("WriteFile(patch append)")};
        }
        if (written != src.size()) {
            return std::unexpected{"WriteFile(patch append): short write"};
        }
        // Flush to disk so the read view (when remapped) can see it. Async flush is the cache's job.
        FlushFileBuffers(active.file);
#else
        if (::pwrite(active.fd, src.data(), src.size(), static_cast<off_t>(offset)) != static_cast<ssize_t>(src.size())) {
            return std::unexpected{std::string{"pwrite(patch) failed: "} + std::strerror(errno)};
        }
        ::fsync(active.fd);
#endif
        active.write_offset += src.size();

        // Remap the read view to include the newly-appended bytes. Cheap: kernel reuses pages.
#if defined(_WIN32)
        if (active.view) UnmapViewOfFile(active.view);
        if (active.mapping) CloseHandle(active.mapping);
        active.view = nullptr;
        active.mapping = nullptr;
        active.mapping = CreateFileMappingW(active.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!active.mapping) return std::unexpected{last_win32_error("CreateFileMapping(patch remap)")};
        void* view = MapViewOfFile(active.mapping, FILE_MAP_READ, 0, 0, 0);
        if (!view) return std::unexpected{last_win32_error("MapViewOfFile(patch remap)")};
        active.view = static_cast<const std::byte*>(view);
        active.view_size = active.write_offset;
#else
        if (active.view) ::munmap(const_cast<std::byte*>(active.view), active.view_size);
        void* view = ::mmap(nullptr, active.write_offset, PROT_READ, MAP_SHARED, active.fd, 0);
        if (view == MAP_FAILED) return std::unexpected{"mmap(patch remap) failed"};
        active.view = static_cast<const std::byte*>(view);
        active.view_size = active.write_offset;
#endif

        const std::uint32_t file_id = static_cast<std::uint32_t>(patch_segments.size()); // 1-based
        return IndexEntry{
            .file_id = file_id,
            .offset = offset,
            .size = static_cast<std::uint32_t>(src.size()),
            .version = new_version,
        };
    }

    std::expected<void, std::string> BlockStore::Impl::persist_index() {
        // Atomic: write to .tmp then rename.
        const auto tmp = dir / "index.bin.tmp";
        const auto final_path = dir / kIndexFile;
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) return std::unexpected{"persist_index: open tmp failed"};
            std::scoped_lock lk(index_mutex);
            out.write(reinterpret_cast<const char*>(index.data()),
                      static_cast<std::streamsize>(index.size() * sizeof(IndexEntry)));
            if (!out) return std::unexpected{"persist_index: write failed"};
        }
        std::error_code ec;
        std::filesystem::rename(tmp, final_path, ec);
        if (ec) return std::unexpected{std::string{"persist_index: rename failed: "} + ec.message()};
        return {};
    }

    void BlockStore::Impl::unmap_all() {
#if defined(_WIN32)
        for (auto& seg : patch_segments) {
            if (seg.view) UnmapViewOfFile(seg.view);
            if (seg.mapping) CloseHandle(seg.mapping);
            if (seg.file != INVALID_HANDLE_VALUE) CloseHandle(seg.file);
            seg = {};
        }
        if (base_view) {
            UnmapViewOfFile(base_view);
            base_view = nullptr;
        }
        if (base_mapping) {
            CloseHandle(base_mapping);
            base_mapping = nullptr;
        }
        if (base_file != INVALID_HANDLE_VALUE) {
            CloseHandle(base_file);
            base_file = INVALID_HANDLE_VALUE;
        }
#else
        for (auto& seg : patch_segments) {
            if (seg.view) ::munmap(const_cast<std::byte*>(seg.view), seg.view_size);
            if (seg.fd >= 0) ::close(seg.fd);
            seg = {};
        }
        if (base_view) {
            ::munmap(const_cast<std::byte*>(base_view), base_view_size);
            base_view = nullptr;
        }
        if (base_fd >= 0) {
            ::close(base_fd);
            base_fd = -1;
        }
#endif
        patch_segments.clear();
    }

} // namespace lfs::core
