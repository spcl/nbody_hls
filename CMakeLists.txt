# Author:  Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Created: January 2018

cmake_minimum_required(VERSION 2.8)
project(nbody_hls C CXX)

# Target options
set(NBODY_PART_NAME "xcku115-flvb2104-2-e" CACHE STRING "Part name for HLS.")
set(NBODY_DSA_NAME "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0" CACHE STRING
    "DSA string for xocc.")
set(NBODY_TARGET_CLOCK 250 CACHE STRING "Target clock for kernel (<=0 uses DSA default).")
set(NBODY_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")

# Domain options
set(NBODY_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(NBODY_KERNEL_WIDTH 4 CACHE STRING "Kernel data path width in number of operands.")
set(NBODY_STEPS 16 CACHE STRING "Number of timesteps.")
set(NBODY_DIMS 3 CACHE STRING "Number of dimensions.")
set(NBODY_TIMESTEP 0.01 CACHE STRING "Size of each timestep.")
set(NBODY_SIZE 1024 CACHE STRING "Number of bodies.")
set(NBODY_TILE_SIZE 4 CACHE STRING "Tile size.")
set(NBODY_SOFTENING_EPSILON 0.00125 CACHE STRING "Epsilon used for Softening.")
set(NBODY_DAMPING_FACTOR 0.995 CACHE STRING "Damping factor for velocity.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(SDAccel REQUIRED)
find_package(Threads REQUIRED)

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${SDAccel_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

configure_file(include/Config.h.in Config.h)

enable_testing()

if(NOT NBODY_USE_SIMPLE)
  set(NBODY_KERNEL_SRC
      ${CMAKE_SOURCE_DIR}/kernel/NBody.cpp
      ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
else()
  set(NBODY_KERNEL_SRC
      ${CMAKE_SOURCE_DIR}/kernel/NBody.cpp)
endif()
add_library(nbodykernel ${NBODY_KERNEL_SRC})
target_link_libraries(nbodykernel ${CMAKE_THREAD_LIBS_INIT})

add_executable(Stats src/Stats.cpp)

add_executable(TestNBody test/TestNBody.cpp)
target_link_libraries(TestNBody ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} nbodykernel)
add_test(TestNBody TestNBody)

add_executable(TestNonDivisiblePack test/TestNonDivisiblePack.cpp)
target_link_libraries(TestNonDivisiblePack ${SDAccel_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} nbodykernel)
add_test(TestNonDivisiblePack TestNonDivisiblePack)

set(NBODY_PROJECT_NAME "NBody")
string(REPLACE ";" " " NBODY_SRC_SYNTHESIS "${NBODY_KERNEL_SRC}")
set(NBODY_ENTRY_FUNCTION "NBody")
if(NBODY_TARGET_CLOCK GREATER 0)
  set(NBODY_TARGET_TIMING "${NBODY_TARGET_CLOCK}")
else()
  set(NBODY_TARGET_TIMING "5")
endif()
configure_file(scripts/Synthesis.tcl.in SynthesizeNBody.tcl)
add_custom_target(synthesize_nbody
  COMMAND ${SDAccel_VIVADO_HLS} -f SynthesizeNBody.tcl)
if(SDAccel_FOUND)
  set(NBODY_XOCC_COMMAND
    -t hw
    -O3
    -s
    -o NBody.xclbin
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_SOURCE_DIR}/hlslib/include
    -I${CMAKE_BINARY_DIR}
    ${NBODY_KERNEL_SRC}
    --kernel NBody
    --xdevice ${NBODY_DSA_NAME}
    --xp prop:kernel.${NBODY_ENTRY_FUNCTION}.kernel_flags="${CMAKE_CXX_FLAGS} -DNBODY_SYNTHESIS -DHLSLIB_SYNTHESIS"
    # Matrices A and C use the same interface
    --xp misc:map_connect=add.kernel.${NBODY_ENTRY_FUNCTION}_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI
    --xp misc:map_connect=add.kernel.${NBODY_ENTRY_FUNCTION}_1.M_AXI_GMEM1.core.OCL_REGION_0.M00_AXI
    --xp misc:map_connect=add.kernel.${NBODY_ENTRY_FUNCTION}_1.M_AXI_GMEM2.core.OCL_REGION_0.M01_AXI
    --max_memory_ports all)
  if(NBODY_TARGET_CLOCK GREATER 0)
    set(NBODY_XOCC_COMMAND ${NBODY_XOCC_COMMAND}
        --kernel_frequency ${NBODY_TARGET_CLOCK})
  endif()
  add_custom_target(kernel_nbody
    COMMAND ${SDAccel_XOCC} ${NBODY_XOCC_COMMAND})
  add_executable(RunNBody.exe host/RunNBody.cpp)
  target_link_libraries(RunNBody.exe ${SDAccel_LIBRARIES} nbodykernel)
endif()
