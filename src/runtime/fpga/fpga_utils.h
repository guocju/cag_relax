/*
 * This file is part of the Xilinx DMA IP Core driver tools for Linux
 *
 * Copyright (c) 2016-present,  Xilinx, Inc.
 * All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */
#include <fcntl.h>  
#include <errno.h> 
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/types.h>

/*
 * man 2 write:
 * On Linux, write() (and similar system calls) will transfer at most
 * 	0x7ffff000 (2,147,479,552) bytes, returning the number of bytes
 *	actually transferred.  (This is true on both 32-bit and 64-bit
 *	systems.)
 */

#define RW_MAX_SIZE	0x7ffff000
#define HOST_TO_DEVICE "/dev/xdma0_h2c_0"
#define DEVICE_TO_HOST "/dev/xdma0_c2h_0"

int verbose = 0;

uint64_t getopt_integer(char *optarg)
{
	int rc;
	uint64_t value;

	rc = sscanf(optarg, "0x%lx", &value);
	if (rc <= 0)
		rc = sscanf(optarg, "%lu", &value);
	//printf("sscanf() = %d, value = 0x%lx\n", rc, value);

	return value;
}

ssize_t read_to_buffer(char *fname, int fd, char *buffer, uint64_t size,
			uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;
	int loop = 0;

	while (count < size) {
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

		if (offset) {
			rc = lseek(fd, offset, SEEK_SET);
			if (rc != offset) {
				fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
					fname, rc, offset);
				perror("seek file");
				return -EIO;
			}
		}

		/* read data from file into memory buffer */
		rc = read(fd, buf, bytes);
		if (rc < 0) {
			fprintf(stderr, "%s, read 0x%lx @ 0x%lx failed %ld.\n",
				fname, bytes, offset, rc);
			perror("read file");
			return -EIO;
		}

		count += rc;
		if (rc != bytes) {
			fprintf(stderr, "%s, read underflow 0x%lx/0x%lx @ 0x%lx.\n",
				fname, rc, bytes, offset);
			break;
		}

		buf += bytes;
		offset += bytes;
		loop++;
	}

	if (count != size && loop)
		fprintf(stderr, "%s, read underflow 0x%lx/0x%lx.\n",
			fname, count, size);
	return count;
}

ssize_t write_from_buffer(char *fname, int fd, char *buffer, size_t size,
			uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;
	int loop = 0;

	while (count < size) {
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

		if (offset) {
			rc = lseek(fd, offset, SEEK_SET);
			if (rc != offset) {
				fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
					fname, rc, offset);
				perror("seek file");
				return -EIO;
			}
		}

		/* write data to file from memory buffer */
		rc = write(fd, buf, bytes);
		if (rc < 0) {
			fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n",
				fname, bytes, offset, rc);
			perror("write file");
			return -EIO;
		}

		count += rc;
		if (rc != bytes) {
			fprintf(stderr, "%s, write underflow 0x%lx/0x%lx @ 0x%lx.\n",
				fname, rc, bytes, offset);
			break;
		}
		buf += bytes;
		offset += bytes;

		loop++;
	}	

	if (count != size && loop)
		fprintf(stderr, "%s, write underflow 0x%lx/0x%lx.\n",
			fname, count, size);

	return count;
}


/* Subtract timespec t2 from t1
 *
 * Both t1 and t2 must already be normalized
 * i.e. 0 <= nsec < 1000000000
 */
static int timespec_check(struct timespec *t)
{
	if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
		return -1;
	return 0;

}

void timespec_sub(struct timespec *t1, struct timespec *t2)
{
	if (timespec_check(t1) < 0) {
		fprintf(stderr, "invalid time #1: %lld.%.9ld.\n",
			(long long)t1->tv_sec, t1->tv_nsec);
		return;
	}
	if (timespec_check(t2) < 0) {
		fprintf(stderr, "invalid time #2: %lld.%.9ld.\n",
			(long long)t2->tv_sec, t2->tv_nsec);
		return;
	}
	t1->tv_sec -= t2->tv_sec;
	t1->tv_nsec -= t2->tv_nsec;
	if (t1->tv_nsec >= 1000000000) {
		t1->tv_sec++;
		t1->tv_nsec -= 1000000000;
	} else if (t1->tv_nsec < 0) {
		t1->tv_sec--;
		t1->tv_nsec += 1000000000;
	}
}

enum fpgaMemcpyKind {
    fpgaMemcpyHostToDevice,
    fpgaMemcpyDeviceToHost,
    fpgaMemcpyDeviceToDevice
};

int fpgaMemcpy(void* to, const void* from, size_t size, fpgaMemcpyKind kind)
{
  ssize_t rc;
  size_t bytes_done = 0;
  int underflow = 0;
  char* devname;
  int fpga_fd;

  	if (kind==fpgaMemcpyHostToDevice)
    {
		devname = HOST_TO_DEVICE;
      fpga_fd = open(devname, O_RDWR);

      if (fpga_fd < 0) {
        fprintf(stderr, "unable to open device %s, %d.\n",
          devname, fpga_fd);
        perror("open device");
        return -EINVAL;
      }

      uint64_t addr = reinterpret_cast<uint64_t>(to);
      rc = write_from_buffer(devname, fpga_fd, (char*)from, size, addr);
      if (rc < 0)
          goto out;
      bytes_done = rc;
      if (bytes_done < size) {
        printf("# underflow %ld/%ld.\n",
           bytes_done, size);
        underflow = 1;
      }
	  goto out;
    }
	else if (kind==fpgaMemcpyDeviceToHost)
	{
		devname = DEVICE_TO_HOST;
      fpga_fd = open(devname, O_RDWR);

      if (fpga_fd < 0) {
        fprintf(stderr, "unable to open device %s, %d.\n",
          devname, fpga_fd);
        perror("open device");
        return -EINVAL;
      }

      uint64_t addr = reinterpret_cast<uint64_t>(from);
      rc = read_to_buffer(devname, fpga_fd, (char*)to, size, addr);
      if (rc < 0)
          goto out;
      bytes_done = rc;
      if (bytes_done < size) {
        printf("# underflow %ld/%ld.\n",
           bytes_done, size);
        underflow = 1;
      }
	  goto out;
	}
	

out:
    close(fpga_fd);
	if (rc < 0)
      return rc;
    /* treat underflow as error */
    return underflow ? -EIO : 0;
}