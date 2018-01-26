/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include "Config.h"
#include "hlslib/DataPack.h"

// constexpr int kMemoryWidth = kMemoryWidthBytes / sizeof(Data_t);
// constexpr int kMemoryWidthVec = kMemoryWidthBytes / sizeof(Vec_t);
// static_assert(kMemoryWidthBytes % sizeof(Data_t) == 0,
//               "Memory width not divisable by size of data type.");
// constexpr int kKernelPerMemory = kMemoryWidth / kKernelWidth;
// static_assert(kMemoryWidth % (kkKernelWidth == 0,
//               "Memory width must be divisable by kernel width.");
// using KernelPack_t = hlslib::DataPack<Data_t, kKernelWidth>;
// using MemoryPack_t = hlslib::DataPack<KernelPack_t, kKernelPerMemory>;
//
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

void NBody(Data_t const mass[], Vec_t const positionIn[], Vec_t positionOut[],
           Vec_t const velocityIn[], Vec_t velocityOut[]);

}

inline Data_t ComputeAcceleration(Data_t const &m1, Data_t const &s0,
                           Data_t const &s1) {
  const auto diff = (s1 - s0);
  return m1 / (diff * diff);
}
