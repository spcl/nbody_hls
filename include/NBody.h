/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "Config.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Operators.h"
#include "hlslib/xilinx/Resource.h"
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/TreeReduce.h"
#include "hlslib/xilinx/Utility.h"

constexpr int kMemoryWidth = kMemoryWidthBytes / sizeof(Data_t);

using MemoryPack_t = hlslib::DataPack<Data_t, kMemoryWidth>;
using Vec_t = hlslib::DataPack<Data_t, kDims>;
using PosMass_t = hlslib::DataPack<Data_t, kDims + 1>;

// How many 512-bit reads/writes are issued for loading in the entire spatial
// dimension (kNBodies * kDims elements)
constexpr unsigned kMemorySizePosition =
    ((kDims + 1) * kNBodies) / kMemoryWidth;
static_assert((kDims * kNBodies) % kMemoryWidth == 0,
              "Position domain size not divisible by memory width.");

constexpr unsigned kMemorySizeVelocity = ((kDims)*kNBodies) / kMemoryWidth;
static_assert((kDims * kNBodies) % kMemoryWidth == 0,
              "Velocity domain size not divisible by memory width.");

constexpr unsigned kNTiles = kNBodies / (kUnrollDepth * kPipelineFactor);
static_assert(kNBodies % (kUnrollDepth * kPipelineFactor) == 0,
              "Domain size not divisible by tile size.");

constexpr unsigned kVectorsPerMemory = kMemoryWidth / (kDims + 1);
static_assert(kMemoryWidth % (kDims + 1) == 0,
              "Memory width must be divisible by dimensions + 1.");

// The number of bodies used as padding to make sure all the writes get pushed
// through. BE CAREFUL: DO NOT USE THE LAST flushfactor ELEMENTS
constexpr int kFlushFactor = 256;
static_assert(kNBodies > kFlushFactor, "Number of bodies too low");

inline double NumberOfOps(const int timesteps) {
  return (6 * kDims + 4.) * static_cast<double>(kNBodies - kFlushFactor) *
             (kNBodies - kFlushFactor) * timesteps +
         (4 * kDims) * (kNBodies - kFlushFactor) * timesteps;
}

extern "C" {

// void NBody(MemoryPack_t const positionMassIn[], MemoryPack_t
// positionMassOut[],
//            Vec_t const velocityIn[], Vec_t velocityOut[]);

void NBody(unsigned timesteps, MemoryPack_t const positionMassIn[],
           MemoryPack_t positionMassOut[], MemoryPack_t const velocityIn[],
           MemoryPack_t velocityOut[],
           hlslib::Stream<MemoryPack_t> &velocityReadMemory,
           hlslib::Stream<Vec_t> &velocityReadKernel,
           hlslib::Stream<MemoryPack_t> &positionMassReadMemory,
           hlslib::Stream<PosMass_t> &positionMassReadKernel,
           hlslib::Stream<Vec_t> &velocityWriteKernel,
           hlslib::Stream<MemoryPack_t> &velocityWriteMemory,
           hlslib::Stream<PosMass_t> &positionMassWriteKernel,
           hlslib::Stream<MemoryPack_t> &positionMassWriteMemory);
}

template <bool soften>
inline Vec_t ComputeAcceleration(PosMass_t const &s0, PosMass_t const &s1) {
  #pragma HLS INLINE
  Data_t diff[kDims];
  Data_t diffSquared[kDims];
  for (int d = 0; d < kDims; ++d) {
    #pragma HLS UNROLL
    const auto diff_i = s1[d] - s0[d];
    HLSLIB_RESOURCE_PRAGMA(diff_i, NBODY_ADD_CORE)
    diff[d] = diff_i;
    const auto squared = diff_i * diff_i;
    HLSLIB_RESOURCE_PRAGMA(squared, NBODY_MULT_CORE)
    diffSquared[d] = squared;
  }
  const Data_t distSquared =
      hlslib::TreeReduce<Data_t, hlslib::op::Add<Data_t>, kDims>(diffSquared);
  const Data_t distSoftened = soften ? (distSquared + kSoftening) : distSquared;
  HLSLIB_RESOURCE_PRAGMA(distSoftened, NBODY_ADD_CORE)
  const Data_t distCubedTmp = distSoftened * distSoftened;
  HLSLIB_RESOURCE_PRAGMA(distCubedTmp, NBODY_MULT_CORE)
  const Data_t distCubed = distCubedTmp * distSoftened;
  HLSLIB_RESOURCE_PRAGMA(distCubed, NBODY_MULT_CORE)
  const Data_t dist = std::sqrt(distCubed);
  const Data_t distCubedReciprocal = Data_t(1) / dist;
  Vec_t acc;
  for (int d = 0; d < kDims; ++d) {
    #pragma HLS UNROLL
    const auto tmp0 = s1[kDims] * diff[d];
    HLSLIB_RESOURCE_PRAGMA(tmp0, NBODY_MULT_CORE)
    const auto tmp1 = tmp0 * distCubedReciprocal;
    HLSLIB_RESOURCE_PRAGMA(tmp1, NBODY_MULT_CORE)
    acc[d] = tmp1;
  }
  return acc;
}
