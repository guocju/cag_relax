# FindP2P.cmake - 查找用户自定义 P2P Runtime

set(__p2p_sdk "/usr/local/p2p")

if(IS_DIRECTORY ${__p2p_sdk})
  message(STATUS "Using P2P SDK at: ${__p2p_sdk}")
  set(P2P_INCLUDE_DIRS ${__p2p_sdk}/include)
  find_library(P2P_DESC_LIBRARY p2p_runtime ${__p2p_sdk}/build)

  if (P2P_DESC_LIBRARY)
    set(P2P_FOUND TRUE)
  endif()
endif()

if (P2P_FOUND)
  message(STATUS "Found P2P_INCLUDE_DIRS = ${P2P_INCLUDE_DIRS}")
  message(STATUS "Found P2P_GPUDRV_LIBRARY = ${P2P_DESC_LIBRARY}")
endif()
