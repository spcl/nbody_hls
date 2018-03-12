/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "NBody.h"
#include "hlslib/Stream.h"


void ReadMemory_PositionMass(MemoryPack_t const memory[],
                             hlslib::Stream<MemoryPack_t> &pipe);

void ContractWidth_PositionMass(hlslib::Stream<MemoryPack_t> &wide,
                                hlslib::Stream<PosMass_t> &narrow);

void ExpandWidth_PositionMass(hlslib::Stream<PosMass_t> &narrow,
                              hlslib::Stream<MemoryPack_t> &wide);

void WriteMemory_PositionMass(hlslib::Stream<MemoryPack_t> &pipe,
                              MemoryPack_t memory[]);

void RepeatFirstTile(hlslib::Stream<PosMass_t> &streamIn,
                     hlslib::Stream<PosMass_t> &streamOut);

void ReadMemory_Velocity(MemoryPack_t const memory[],
                         hlslib::Stream<MemoryPack_t> &pipe);

// Used for testing software. Does not work with AXI if kDims is 3 
void ReadSingle_Velocity(Vec_t const memory[],
                         hlslib::Stream<Vec_t> &pipe);

void WriteMemory_Velocity(hlslib::Stream<MemoryPack_t> &pipe,
                          MemoryPack_t memory[]);

// Used for testing software. Does not work with AXI if kDims is 3 
void WriteSingle_Velocity(hlslib::Stream<Vec_t> &pipe,
                          Vec_t memory[]);

#ifndef HLSLIB_SYNTHESIS

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
template <typename T, unsigned width>
void ConvertMemoryToNonDivisible(
    hlslib::Stream<MemoryPack_t> &streamIn,
    hlslib::Stream<hlslib::DataPack<T, width>> &streamOut, int iterations) {
  constexpr unsigned kOutBytes = sizeof(T) * width;
  constexpr unsigned kOutBits = 8 * kOutBytes;
  ap_uint<6> bytesRemaining = 0;
  MemoryPack_t curr;
  MemoryPack_t next;
  ap_uint<kOutBits> out;
  for (unsigned i = 0; i < iterations; ++i) {
    #pragma HLS PIPELINE II=1
    if (bytesRemaining == 0) {
      curr = streamIn.Pop();
      out = curr.data().range(kOutBits - 1, 0);
      bytesRemaining = bytesRemaining - kOutBytes;
    } else if (bytesRemaining < kOutBytes) {
      next = streamIn.Pop();
      out.range(8 * bytesRemaining - 1, 0) = curr.data().range(
          8 * kMemoryWidthBytes - 1, 8 * (kMemoryWidthBytes - bytesRemaining));
      out.range(kOutBits - 1, 8 * bytesRemaining) =
          next.data().range(8 * (kOutBytes - bytesRemaining) - 1, 0);
      bytesRemaining = kMemoryWidthBytes - (kOutBytes - bytesRemaining);
      curr = next;
    } else { // bytesRemaining >= kOutBytes
      out = curr.data().range(
          8 * (kMemoryWidthBytes + kOutBytes - bytesRemaining) - 1,
          8 * (kMemoryWidthBytes - bytesRemaining));
      bytesRemaining = bytesRemaining - kOutBytes;
    }
    streamOut.Push(*reinterpret_cast<T const *>(&out));
  }
}

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
template <typename T, unsigned width>
void ConvertNonDivisibleToMemory(
    hlslib::Stream<hlslib::DataPack<T, width>> &streamIn,
    hlslib::Stream<MemoryPack_t> &streamOut, unsigned iterations) {
  constexpr unsigned kInBytes = sizeof(T) * width;
  constexpr unsigned kInBits = 8 * kInBytes;
  ap_uint<hlslib::ConstLog2(kMemoryWidthBytes)> bytesMissing =
      kMemoryWidthBytes;
  MemoryPack_t out;
  for (unsigned i = 0; i < iterations; ++i) {
    #pragma HLS PIPELINE II=1
    const auto readT = streamIn.Pop();
    const auto read = *reinterpret_cast<ap_uint<kInBits> const *>(&readT);
    if (bytesMissing == kInBytes) {
      out.data().range(8 * kMemoryWidthBytes - 1,
                       8 * (kMemoryWidthBytes - kInBytes)) = read;
      bytesMissing = 64;
      streamOut.Push(out);
    } else if (bytesMissing < kInBytes) {
      out.data().range(8 * kMemoryWidthBytes - 1,
                       8 * (kMemoryWidthBytes - bytesMissing)) =
          read.range(8 * bytesMissing - 1, 0);
      MemoryPack_t next;
      next.data().range(8 * (kInBytes - bytesMissing) - 1, 0) =
          read.range(kInBits - 1, 8 * bytesMissing);
      streamOut.Push(out);
      out = next;
      bytesMissing = 64 - kInBytes + bytesMissing;
    } else {  // bytesMissing > kInBytes 
      out.data().range(8 * (kMemoryWidthBytes - bytesMissing + kInBytes) - 1,
                       8 * (kMemoryWidthBytes - bytesMissing)) = read;
      bytesMissing = bytesMissing - kInBytes;
    }
  }
}

#endif
