if(USE_FPGA)
  # # 启用 CUDA，防止头文件触发错误探测
  # enable_language(CUDA)
  # set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
  # set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # 你也可以写多个架构

  # 确保能找到 FindP2P.cmake
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils)
  find_package(P2P REQUIRED)

  if (NOT P2P_FOUND)
    message(FATAL_ERROR "Cannot find P2P runtime in /usr/local/p2p")
  endif()

  message(STATUS "Build with P2P runtime support")

  # 添加 include 路径
  include_directories(SYSTEM ${P2P_INCLUDE_DIRS})

  # 链接库
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${P2P_DESC_LIBRARY})
  # Add P2P builtins to RelaxVM
  tvm_file_glob(GLOB RELAX_VM_P2P_BUILTIN_SRC_CC src/runtime/relax_vm/p2p/*.cc)
  list(APPEND RUNTIME_SRCS ${RELAX_VM_P2P_BUILTIN_SRC_CC})
endif()