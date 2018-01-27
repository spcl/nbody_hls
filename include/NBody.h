/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "Config.h"
#include "hlslib/DataPack.h"
#include "hlslib/Operators.h"
#include "hlslib/TreeReduce.h"
#include "hlslib/Utility.h"

constexpr int kMemoryWidth = kMemoryWidthBytes / sizeof(Data_t);
// constexpr int kMemoryWidthVec = kMemoryWidthBytes / sizeof(Vec_t);
// static_assert(kMemoryWidthBytes % sizeof(Data_t) == 0,
//               "Memory width not divisable by size of data type.");
// constexpr int kKernelPerMemory = kMemoryWidth / kKernelWidth;
// static_assert(kMemoryWidth % (kkKernelWidth == 0,
//               "Memory width must be divisable by kernel width.");
// using KernelPack_t = hlslib::DataPack<Data_t, kKernelWidth>;
using MemoryPack_t = hlslib::DataPack<Data_t, kMemoryWidth>;
using Vec_t = hlslib::DataPack<Data_t, kDims>;

// constexpr int kNMemory = kN / kMemoryWidth;
// static_assert(kN % kMemoryWidth == 0,
//               "N must be divisable by memory width.");

// constexpr int kTileSizeMemory = kTileSize / kMemoryWidth;
// static_assert(kTileSize % kMemoryWidth == 0,
//               "Tile size must be divisable by memory width");

struct Packed {
  Vec_t acc;
  Vec_t pos;
  Data_t mass;
  Packed(Vec_t const &_acc, Vec_t const &_pos, Data_t const &_mass)
      : acc(_acc), pos(_pos), mass(_mass) {}
  Packed() = default;
  Packed(Packed const &) = default;
  Packed(Packed &&) = default;
  ~Packed() = default;
  Packed &operator=(Packed const &) = default;
  Packed &operator=(Packed &&) = default;
};

extern "C" {

void NBody(MemoryPack_t const mass[], Vec_t const positionIn[],
           Vec_t positionOut[], Vec_t const velocityIn[], Vec_t velocityOut[]);
}

inline Vec_t ComputeAcceleration(Data_t const &m1, Vec_t const &s0,
                                 Vec_t const &s1) {
  #pragma HLS INLINE
  Data_t diff[kDims];
  Data_t diffSquared[kDims];
  for (int d = 0; d < kDims; ++d) {
    #pragma HLS UNROLL
    const auto diff_i = s1[d] - s0[d];
    diff[d] = diff_i;
    diffSquared[d] = diff_i * diff_i;
  }
  const Data_t distSquared =
      hlslib::TreeReduce<Data_t, hlslib::op::Add<Data_t>, kDims>(diffSquared);
  const Data_t dist = std::sqrt(distSquared);
  const Data_t distCubed = dist * dist * dist;
  const Data_t distCubedReciprocal = Data_t(1) / distCubed;
  Vec_t acc;
  for (int d = 0; d < kDims; ++d) {
    #pragma HLS UNROLL
    acc[d] = m1 * diff[d] * distCubedReciprocal; 
  }
  return acc;
}
