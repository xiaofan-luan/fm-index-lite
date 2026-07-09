// Licensed to the LF AI & Data foundation under Apache-2.0.
// POSIX file serdes for FMIndex: write a serialized index to a file, and open
// it zero-copy via mmap (the index views the mapping). This is a convenience /
// demo layer; the Milvus port uses Milvus's own storage + mmap infrastructure
// and only needs FMIndex::Serialize / FMIndex::LoadView.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "index/fmindex/FMIndex.h"

namespace milvus::index::fmindex {

// Streams the index straight to the file (no intermediate full-index blob).
inline bool
SaveToFile(const FMIndex& fm, const std::string& path) {
    return fm.SerializeToFile(path);
}

// An FMIndex backed by a read-only memory mapping. The mapping (page-aligned, so
// the 8-byte payload alignment holds) stays alive for this object's lifetime and
// the index views it zero-copy — serving RAM is just the rebuilt directories.
class MappedFMIndex {
 public:
    static std::unique_ptr<MappedFMIndex>
    Open(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return nullptr;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            return nullptr;
        }
        size_t sz = static_cast<size_t>(st.st_size);
        void* addr = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            return nullptr;
        }
        std::unique_ptr<MappedFMIndex> self(new MappedFMIndex());
        self->fd_ = fd;
        self->addr_ = addr;
        self->size_ = sz;
        self->index_ = FMIndex::LoadView(
            reinterpret_cast<const uint8_t*>(addr), sz);
        return self;
    }

    ~MappedFMIndex() {
        if (addr_ && addr_ != MAP_FAILED) {
            ::munmap(addr_, size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    MappedFMIndex(const MappedFMIndex&) = delete;
    MappedFMIndex&
    operator=(const MappedFMIndex&) = delete;

    const FMIndex&
    index() const {
        return index_;
    }

 private:
    MappedFMIndex() = default;
    int fd_ = -1;
    void* addr_ = nullptr;
    size_t size_ = 0;
    FMIndex index_;
};

}  // namespace milvus::index::fmindex
