#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include "fpga_common.h"

namespace tvm {
namespace runtime {

class FPGADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { FPGA_CALL(fpgaSetDevice(dev.device_id)); }

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final { return; }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    ICHECK_EQ(64 % alignment, 0U) << "FPGA space is aligned at 64 bytes";
    void* ret;
    FPGA_CALL(fpgaSetDevice(dev.device_id));
    size_t free_mem, total_mem;
    FPGA_CALL(fpgaMemGetInfo(&free_mem, &total_mem));
    VLOG(1) << "allocating " << nbytes << " bytes on device, with " << free_mem
            << " bytes currently free out of " << total_mem << " bytes available";
    FPGA_CALL(fpgaMalloc(ret, nbytes));
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    FPGA_CALL(fpgaSetDevice(dev.device_id));
    VLOG(1) << "freeing device memory";
    FPGA_CALL(fpgaFree(ptr));
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }
    if (dev_from.device_type == kDLFPGA && dev_to.device_type == kDLCPU) {
      FPGA_CALL(fpgaSetDevice(dev_from.device_id));
      FPGACopy(from, to, size, fpgaMemcpyDeviceToHost, stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLFPGA) {
      FPGA_CALL(fpgaSetDevice(dev_to.device_id));
      FPGACopy(from, to, size, fpgaMemcpyHostToDevice, stream);
    } else if (dev_from.device_type == kDLFPGA && dev_to.device_type == kDLCUDA) {
      FPGA_CALL(fpgaSetDevice(dev_to.device_id));
      FPGACopy(from, to, size, fpgaMemcpyToGPU, stream);
    } else {
      LOG(FATAL) << "expect copy from/to FPGA or in FPGA";
    }
  }

  static FPGADeviceAPI* Global() {
    static FPGADeviceAPI* inst = new FPGADeviceAPI();
    return inst;
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final { return; }

 private:
  static void FPGACopy(const void* from, void* to, size_t size, fpgaMemcpyKind kind,
                       TVMStreamHandle stream) {
    if (stream != nullptr) {
      // FPGA_CALL(fpgaMemcpyAsync(to, from, size, kind, stream));
    } else {
      FPGA_CALL(fpgaMemcpy(to, from, size, kind));
    }
  }
};

TVM_REGISTER_GLOBAL("device_api.fpga").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = FPGADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm