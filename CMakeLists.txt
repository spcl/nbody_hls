# Author:  Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Created: January 2018

cmake_minimum_required(VERSION 2.8)
project(nbody_hls C CXX)

# Target options
set(NBODY_PART_NAME "xcku115-flvb2104-2-e" CACHE STRING "Part name for HLS.")
set(NBODY_DSA_NAME "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0" CACHE STRING
    "DSA string for xocc.")
set(NBODY_TARGET_CLOCK 300 CACHE STRING "Target clock for kernel.")
set(NBODY_TARGET_CLOCK_UNCERTAINTY 1.08 CACHE STRING "Clock uncertainty for HLS.")
set(NBODY_FLATTEN OFF CACHE STRING "Flatten the loops to save cycles on saturating and draining the pipeline")
set(NBODY_SINGLE_SLR OFF CACHE STRING "Only place and route on a single SLR of the KU115")
set(NBODY_TMP_DIR /tmp CACHE STRING "Temporary directory for SDx build files")

# Domain options
set(NBODY_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(NBODY_N 512 CACHE STRING "Number of bodies.")
set(NBODY_STEPS 16 CACHE STRING "Number of timesteps.")
set(NBODY_DIMS 3 CACHE STRING "Number of dimensions.")
set(NBODY_TIMESTEP 0.01 CACHE STRING "Size of each timestep.")
set(NBODY_UNROLL_DEPTH 4 CACHE STRING "Tile size for depth unrolling.")
set(NBODY_PIPELINE_FACTOR 16 CACHE STRING
    "Factor to increase tile size for pipelining accumulations.")
set(NBODY_SOFTENING_TERM 0.00125 CACHE STRING "Epsilon used for Softening.")
set(NBODY_DAMPING_FACTOR 0.995 CACHE STRING "Damping factor for velocity.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(Vitis REQUIRED)
find_package(Threads REQUIRED)

include_directories(include hlslib/include ${CMAKE_BINARY_DIR} ${Vitis_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -DHLSLIB_LEGACY_SDX=0")
if(NBODY_FLATTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNBODY_FLATTEN")
endif()

configure_file(include/Config.h.in Config.h)

enable_testing()

set(NBODY_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/NBody.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp)
add_library(nbodykernel ${NBODY_KERNEL_SRC})
target_link_libraries(nbodykernel ${CMAKE_THREAD_LIBS_INIT})

add_executable(Stats src/Stats.cpp)
target_link_libraries(Stats ${Vitis_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} nbodykernel)

add_executable(TestNBody test/TestNBody.cpp)
target_link_libraries(TestNBody ${Vitis_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} nbodykernel)
add_test(TestNBody TestNBody)

# HLS setup
set(NBODY_PROJECT_NAME "NBody")
string(REPLACE ";" " " NBODY_SRC_SYNTHESIS "${NBODY_KERNEL_SRC}")
set(NBODY_ENTRY_FUNCTION "NBody")
configure_file(scripts/Synthesis.tcl.in SynthesizeNBody.tcl)
add_custom_target(synthesize_nbody
  COMMAND ${Vitis_VIVADO_HLS} -f SynthesizeNBody.tcl)

# Vivado project options
set(DONUT_KERNEL_NAME nbody_kernel) 
set(DONUT_KERNEL_DIRECTORY ${CMAKE_BINARY_DIR})
set(DONUT_VENDOR_NAME SPCL)
set(DONUT_TMP_DIR ${NBODY_TMP_DIR})
set(DONUT_PART_NAME ${NBODY_PART_NAME})
set(DONUT_SOURCES_DIRECTORY ${CMAKE_BINARY_DIR}/${NBODY_PROJECT_NAME}/${NBODY_PART_NAME}/syn/verilog)
if(Vitis_VERSION STREQUAL "2017.4")
  set(DONUT_CALL_PACKAGE_FUNCTION 1)
else()
  set(DONUT_CALL_PACKAGE_FUNCTION 0)
endif() 
configure_file(scripts/config.tcl.in config.tcl)
configure_file(hw/top.v.in top.v)
configure_file(scripts/O_opt_design_pre.tcl data/sdx/xocc/O_opt_design_pre.tcl) 
configure_file(scripts/optMap.xml data/sdx/xocc/optMap.xml) 

# SDx kernel wizard custom RTL project
add_custom_target(setup_project COMMAND ${CMAKE_SOURCE_DIR}/scripts/project_manager.py setup ${DONUT_KERNEL_NAME} ${Vitis_VIVADO} ${CMAKE_BINARY_DIR} ${DONUT_TMP_DIR})
add_custom_target(package_kernel COMMAND ${CMAKE_SOURCE_DIR}/scripts/project_manager.py package ${DONUT_KERNEL_NAME} ${Vitis_VIVADO} ${CMAKE_BINARY_DIR} ${DONUT_TMP_DIR})

# Remove Vivado project when running clean
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
             ${DONUT_KERNEL_DIRECTORY}/${DONUT_KERNEL_NAME}_ex)

set(NBODY_VPP_COMMAND
  -t hw
  -l
  -O3
  -s
  -o NBody.xclbin
  -I${CMAKE_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/hlslib/include
  -I${CMAKE_BINARY_DIR}
  ${DONUT_KERNEL_NAME}.xo 
  --kernel NBody
  --platform ${NBODY_DSA_NAME}
  --xp prop:kernel.${NBODY_ENTRY_FUNCTION}.kernel_flags="${CMAKE_CXX_FLAGS} -DNBODY_SYNTHESIS -DHLSLIB_SYNTHESIS"
  --sp ${DONUT_KERNEL_NAME}_1.m_axi_gmem0:bank0
  --sp ${DONUT_KERNEL_NAME}_1.m_axi_gmem1:bank1
  --advanced.prop kernel.${DONUT_KERNEL_NAME}.kernel_flags="${CMAKE_CXX_FLAGS} -DNBODY_SYNTHESIS -DHLSLIB_SYNTHESIS")
if(NBODY_SINGLE_SLR)
  set(NBODY_VPP_COMMAND ${NBODY_VPP_COMMAND} -O4)
else()
  set(NBODY_VPP_COMMAND ${NBODY_VPP_COMMAND} -O3)
endif()
if(NBODY_TARGET_CLOCK GREATER 0)
  set(NBODY_VPP_COMMAND ${NBODY_VPP_COMMAND}
      --kernel_frequency ${NBODY_TARGET_CLOCK})
endif()
add_custom_target(build_kernel
  COMMAND XILINX_PATH=${CMAKE_BINARY_DIR} ${Vitis_COMPILER} ${NBODY_VPP_COMMAND})
add_executable(RunNBody.exe host/RunNBody.cpp)
target_link_libraries(RunNBody.exe ${Vitis_LIBRARIES} nbodykernel)
