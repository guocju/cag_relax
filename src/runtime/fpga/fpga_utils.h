#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <vector>

#include "MemAllocator.h"
#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#define RW_MAX_SIZE 0x7ffff000
#define HOST_TO_DEVICE "/dev/xdma0_h2c_0"
#define DEVICE_TO_HOST "/dev/xdma0_c2h_0"
struct gpu_buffer {
  void* address;
  size_t size;
};
static std::vector<gpu_buffer> GpuBufferArray = {};

inline ssize_t read_to_buffer(char* fname, int fd, char* buffer, uint64_t size, uint64_t base) {
  ssize_t rc;
  uint64_t count = 0;
  char* buf = buffer;
  off_t offset = base;
  int loop = 0;

  while (count < size) {
    uint64_t bytes = size - count;

    if (bytes > RW_MAX_SIZE) bytes = RW_MAX_SIZE;

    if (offset) {
      rc = lseek(fd, offset, SEEK_SET);
      if (rc != offset) {
        fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n", fname, rc, offset);
        perror("seek file");
        return -EIO;
      }
    }

    /* read data from file into memory buffer */
    rc = read(fd, buf, bytes);
    if (rc < 0) {
      fprintf(stderr, "%s, read 0x%lx @ 0x%lx failed %ld.\n", fname, bytes, offset, rc);
      perror("read file");
      return -EIO;
    }

    count += rc;
    if (rc != bytes) {
      fprintf(stderr, "%s, read underflow 0x%lx/0x%lx @ 0x%lx.\n", fname, rc, bytes, offset);
      break;
    }

    buf += bytes;
    offset += bytes;
    loop++;
  }

  if (count != size && loop)
    fprintf(stderr, "%s, read underflow 0x%lx/0x%lx.\n", fname, count, size);
  return count;
}

inline ssize_t write_from_buffer(char* fname, int fd, char* buffer, size_t size, uint64_t base) {
  ssize_t rc;
  uint64_t count = 0;
  char* buf = buffer;
  off_t offset = base;
  int loop = 0;

  while (count < size) {
    uint64_t bytes = size - count;

    if (bytes > RW_MAX_SIZE) bytes = RW_MAX_SIZE;

    if (offset) {
      rc = lseek(fd, offset, SEEK_SET);
      if (rc != offset) {
        fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n", fname, rc, offset);
        perror("seek file");
        return -EIO;
      }
    }

    /* write data to file from memory buffer */
    rc = write(fd, buf, bytes);
    if (rc < 0) {
      fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n", fname, bytes, offset, rc);
      perror("write file");
      return -EIO;
    }

    count += rc;
    if (rc != bytes) {
      fprintf(stderr, "%s, write underflow 0x%lx/0x%lx @ 0x%lx.\n", fname, rc, bytes, offset);
      break;
    }
    buf += bytes;
    offset += bytes;

    loop++;
  }

  if (count != size && loop)
    fprintf(stderr, "%s, write underflow 0x%lx/0x%lx.\n", fname, count, size);

  return count;
}

enum fpgaMemcpyKind { fpgaMemcpyHostToDevice, fpgaMemcpyDeviceToHost, fpgaMemcpyToGPU };

inline int fpgaMemcpy(void* to, const void* from, size_t size, fpgaMemcpyKind kind) {
  ssize_t rc;
  size_t bytes_done = 0;
  int underflow = 0;
  char* devname;
  int fpga_fd;

  if (kind == fpgaMemcpyHostToDevice) {
    devname = HOST_TO_DEVICE;
    fpga_fd = open(devname, O_RDWR);

    if (fpga_fd < 0) {
      fprintf(stderr, "unable to open device %s, %d.\n", devname, fpga_fd);
      perror("open device");
      return -EINVAL;
    }

    uint64_t addr = reinterpret_cast<uint64_t>(to);
    rc = write_from_buffer(devname, fpga_fd, (char*)from, size, addr);
    if (rc < 0) goto out;
    bytes_done = rc;
    if (bytes_done < size) {
      printf("# underflow %ld/%ld.\n", bytes_done, size);
      underflow = 1;
    }
    goto out;
  } else if (kind == fpgaMemcpyDeviceToHost) {
    devname = DEVICE_TO_HOST;
    fpga_fd = open(devname, O_RDWR);

    if (fpga_fd < 0) {
      fprintf(stderr, "unable to open device %s, %d.\n", devname, fpga_fd);
      perror("open device");
      return -EINVAL;
    }

    uint64_t addr = reinterpret_cast<uint64_t>(from);
    rc = read_to_buffer(devname, fpga_fd, (char*)to, size, addr);
    if (rc < 0) goto out;
    bytes_done = rc;
    if (bytes_done < size) {
      printf("# underflow %ld/%ld.\n", bytes_done, size);
      underflow = 1;
    }
    goto out;
  } else if (kind == fpgaMemcpyToGPU) {
    gpu_buffer buffer{to, size};
    GpuBufferArray.emplace_back(buffer);
  }

out:
  close(fpga_fd);
  if (rc < 0) return rc;
  /* treat underflow as error */
  return underflow ? -EIO : 0;
}

inline void fpgaMemGetInfo(size_t* free_mem, size_t* total_mem) {
  MemAllocator* allocator = MemAllocator::Global();
  *free_mem = allocator->get_free_mem();
  *total_mem = allocator->get_total_mem();
}

inline void fpgaMalloc(void*& ret, size_t nbytes) {
  MemAllocator* allocator = MemAllocator::Global();
  ret = allocator->allocate(nbytes);
}

inline void fpgaFree(void* ptr) {
  MemAllocator* allocator = MemAllocator::Global();
  allocator->deallocate(ptr);
}

inline void translate_and_transfer() {
  /*
  translate();
  transfer();
  GpuBufferArray = {};
  */
}
inline void launch_p2p_kernel(int slice_number) {}
inline void fpgaModuleLaunchKernel(int data_size, int slice_num, int* buffer_sizes, int param_num,
                                   void** ptrs, int ptr_num, int buffer_num) {
  /*
  transferCtrlKernelCode(code, data, addr);
  TransferParams(ptrs, ptr_num, data_size, buffer_sizes, param_num, buffer_num, slice_num);
  mmap_fpga_register();
  StartCtrlKernel();
  */
}

#endif