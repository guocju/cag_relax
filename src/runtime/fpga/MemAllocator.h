#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

class MemAllocator {
 private:
  static constexpr size_t FPGA_MEMORY_SIZE = 512 * 1024 * 1024 - 16 * 1024;  // 512MB -16KB
  static constexpr size_t BLOCK_SIZE = 64;                                   // 64 bytes
  static constexpr size_t BLOCK_COUNT = FPGA_MEMORY_SIZE / BLOCK_SIZE;

  std::vector<bool> bitmap;
  uint8_t* base_address;
  std::mutex mtx;
  size_t last_alloc_index;
  size_t free_mem;
  size_t* ptr_sizes;

 public:
  MemAllocator() : last_alloc_index(0) {
    free_mem = FPGA_MEMORY_SIZE;
    bitmap.resize(BLOCK_COUNT, false);
    ptr_sizes = new size_t[BLOCK_COUNT]();
    base_address = reinterpret_cast<uint8_t*>(0x20004000);
  }

  ~MemAllocator() { delete[] ptr_sizes; }

  void* allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mtx);
    if (size == 0) {
      throw std::invalid_argument("Size can't be 0");
    }

    size_t blocks_needed = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t start_index = last_alloc_index;
    for (size_t i = 0; i < BLOCK_COUNT; ++i) {
      size_t idx = (start_index + i) % BLOCK_COUNT;
      bool found = true;
      for (size_t j = 0; j < blocks_needed; ++j) {
        if (idx + j >= BLOCK_COUNT || bitmap[idx + j]) {
          found = false;
          break;
        }
      }
      if (found) {
        for (size_t j = 0; j < blocks_needed; ++j) {
          bitmap[idx + j] = true;
        }
        last_alloc_index = (idx + blocks_needed) % BLOCK_COUNT;
        ptr_sizes[idx] = size;
        free_mem -= blocks_needed * BLOCK_SIZE;
        return base_address + idx * BLOCK_SIZE;
      }
    }
    throw std::runtime_error("FPGA memory full");
  }

  void deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mtx);
    size_t start_index = (static_cast<uint8_t*>(ptr) - base_address) / BLOCK_SIZE;
    size_t size = ptr_sizes[start_index];

    size_t blocks_to_free = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (size == 0) {
      throw std::invalid_argument("Pointer not found in allocation map");
    }
    if (start_index + blocks_to_free > BLOCK_COUNT) {
      throw std::invalid_argument("Pointer out of range");
    }

    for (size_t i = 0; i < blocks_to_free; ++i) {
      bitmap[start_index + i] = false;
    }
    ptr_sizes[start_index] = 0;
    free_mem += blocks_to_free * BLOCK_SIZE;
  }

  size_t get_total_mem() const { return FPGA_MEMORY_SIZE; }

  size_t get_free_mem() const { return free_mem; }

  static MemAllocator* Global() {
    static MemAllocator* inst = new MemAllocator();
    return inst;
  }
};