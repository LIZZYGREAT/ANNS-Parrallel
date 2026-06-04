#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <new>
#if !defined(_WIN32)
#include <malloc.h>
#endif

#ifndef ALIGNED_ALLOC_ALIGN
#if defined(__aarch64__) || defined(__ARM_NEON)
constexpr size_t kDefaultAlign = 64;
#elif defined(__AVX512F__)
constexpr size_t kDefaultAlign = 64;
#elif defined(__AVX2__)
constexpr size_t kDefaultAlign = 32;
#else
constexpr size_t kDefaultAlign = 16;
#endif
#else
constexpr size_t kDefaultAlign = ALIGNED_ALLOC_ALIGN;
#endif

inline constexpr size_t simd_min_align() {
#if defined(__aarch64__) || defined(__ARM_NEON)
    return 16;
#elif defined(__AVX2__)
    return 32;
#else
    return 16;
#endif
}

template <typename T, std::size_t Alignment = kDefaultAlign>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) ptr = nullptr;
        if (!ptr) throw std::bad_alloc();
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
    template <typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

template<typename T>
class AlignedBuffer {
    T* ptr_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0; 
    static constexpr size_t kAlign = kDefaultAlign;

    void release() {
        if (ptr_) {
#if defined(_WIN32)
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
            ptr_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
    }

public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) { resize(n); }
    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_), capacity_(o.capacity_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
        o.capacity_ = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            size_ = o.size_;
            capacity_ = o.capacity_;
            o.ptr_ = nullptr;
            o.size_ = 0;
            o.capacity_ = 0;
        }
        return *this;
    }

    void reserve(size_t new_cap) {
        if (new_cap <= capacity_) return;

        void* p = nullptr;
#if defined(_WIN32)
        p = _aligned_malloc(new_cap * sizeof(T), kAlign);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, kAlign, new_cap * sizeof(T)) != 0) p = nullptr;
        if (!p) throw std::bad_alloc();
#endif
        T* new_ptr = static_cast<T*>(p);

        std::memset(new_ptr, 0, new_cap * sizeof(T));

        if (ptr_ && size_ > 0) {
            std::memcpy(new_ptr, ptr_, size_ * sizeof(T));
        }

        release(); 
        ptr_ = new_ptr;
        capacity_ = new_cap;
    }

    void resize(size_t n) {
        if (n > capacity_) {
            size_t next_cap = capacity_ == 0 ? n : capacity_ * 2;
            reserve(std::max(n, next_cap));
        }
        size_ = n;
    }

    void assign(const std::vector<T>& v) {
        resize(v.size());
        if (!v.empty()) {
            std::memcpy(ptr_, v.data(), v.size() * sizeof(T));
        }
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    static constexpr size_t alignment() noexcept { return kAlign; }
};