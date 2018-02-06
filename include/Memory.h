/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "NBody.h"
#include "hlslib/Stream.h"

/// Receives velocities from memory and updates them with acceleration values
/// received from the kernel, and computes the updated positions when a
/// timestep finishes.
void UpdateVelocityAndPosition(hlslib::Stream<Packed> &fromKernel,
                               hlslib::Stream<Vec_t> &velocityIn,
                               hlslib::Stream<Vec_t> &velocityOut,
                               hlslib::Stream<Vec_t> &positionOut);

/// Directly read vectors from memory. Does not work if the vector is not of
/// binary size
void ReadVector(Vec_t const memory[], hlslib::Stream<Vec_t> &stream);

/// Reads 512-bit values from memory, to be converted into vectors by a
/// subsequent module
void ReadVectorMemory(MemoryPack_t const memory[],
                      hlslib::Stream<MemoryPack_t> &stream);

/// Every timestep, buffer and repeat the first tile, as this must be stored in
/// the processing elements before streaming can begin
void RepeatFirstTile(hlslib::Stream<Packed> &streamIn,
                     hlslib::Stream<Packed> &streamOut);

/// Directly writes velocity vectors to memory. Does not work if the vector is
/// not of binary size
void WriteVelocity(hlslib::Stream<Vec_t> &stream, Vec_t memory[]);

/// Directly writes position vectors to memory. Does not work if the vector is
/// not of binary size
void WritePosition(hlslib::Stream<Vec_t> &stream, Vec_t memory[]);

/// Writes velocity vectors to memory after they've been converted to the memory
/// width.
void WriteVelocityMemory(hlslib::Stream<MemoryPack_t> &stream,
                         MemoryPack_t memory[]);

/// Directly writes position vectors to memory. Does not work if the vector is
/// not of binary size
void WritePositionMemory(hlslib::Stream<MemoryPack_t> &stream,
                         MemoryPack_t memory[]);

/// Reads 512-bit wide vectors of mass values to be converted into scalars or
/// shorter vectors by the subsequent module
void ReadMass(MemoryPack_t const memory[],
              hlslib::Stream<MemoryPack_t> &stream);

/// Packs separate position and mass streams into a single packed bus to be
/// passed to the kernel
void PackData(hlslib::Stream<Vec_t> &posIn, hlslib::Stream<Data_t> &massIn,
              hlslib::Stream<Packed> &packedOut);

/// Narrows 512-bit wide vectors from memory into scalar mass elements
void UnpackMass(hlslib::Stream<MemoryPack_t> &streamIn,
                hlslib::Stream<Data_t> &streamOut);

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
void ConvertMemoryToVector(hlslib::Stream<MemoryPack_t> &streamIn,
                           hlslib::Stream<Vec_t> &streamOut);

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
void ConvertVelocityToMemory(hlslib::Stream<Vec_t> &streamIn,
                             hlslib::Stream<MemoryPack_t> &streamOut);

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
void ConvertPositionToMemory(hlslib::Stream<Vec_t> &streamIn,
                             hlslib::Stream<MemoryPack_t> &streamOut);

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
template <unsigned iterations, typename T>
void ConvertMemoryToNonDivisible(hlslib::Stream<MemoryPack_t> &streamIn,
                                 hlslib::Stream<T> &streamOut) {
  ap_uint<6> bytesRemaining = 0;
  MemoryPack_t curr;
  MemoryPack_t next;
  ap_uint<8 * sizeof(T)> out;
  for (unsigned i = 0; i < iterations; ++i) {
    #pragma HLS PIPELINE II=1
    if (bytesRemaining == 0) {
      curr = streamIn.ReadBlocking();
      out = curr.data().range(8 * sizeof(T) - 1, 0);
      bytesRemaining = bytesRemaining - sizeof(T);
    } else if (bytesRemaining < sizeof(T)) {
      next = streamIn.ReadBlocking();
      out.range(8 * bytesRemaining - 1, 0) = curr.data().range(
          8 * kMemoryWidthBytes - 1, 8 * (kMemoryWidthBytes - bytesRemaining));
      out.range(8 * sizeof(T) - 1, 8 * bytesRemaining) =
          next.data().range(8 * (sizeof(T) - bytesRemaining) - 1, 0);
      bytesRemaining = kMemoryWidthBytes - (sizeof(T) - bytesRemaining);
      curr = next;
    } else { // bytesRemaining >= sizeof(T)
      out = curr.data().range(
          8 * (kMemoryWidthBytes + sizeof(T) - bytesRemaining) - 1,
          8 * (kMemoryWidthBytes - bytesRemaining));
      bytesRemaining = bytesRemaining - sizeof(T);
    }
    hlslib::WriteBlocking(streamOut, *reinterpret_cast<T const *>(&out));
  }
}

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
template <unsigned iterations, typename T>
void ConvertNonDivisibleToMemory(hlslib::Stream<T> &streamIn,
                                 hlslib::Stream<MemoryPack_t> &streamOut) {
  ap_uint<hlslib::ConstLog2(kMemoryWidthBytes)> bytesMissing =
      kMemoryWidthBytes;
  MemoryPack_t out;
  for (unsigned i = 0; i < iterations; ++i) {
    #pragma HLS PIPELINE II=1
    const auto readT = streamIn.ReadBlocking();
    const auto read = *reinterpret_cast<ap_uint<8 * sizeof(T)> const *>(&readT);
    if (bytesMissing == sizeof(T)) {
      out.data().range(8 * kMemoryWidthBytes - 1,
                       8 * (kMemoryWidthBytes - sizeof(T))) = read;
      bytesMissing = 64;
      streamOut.WriteBlocking(out);
    } else if (bytesMissing < sizeof(T)) {
      out.data().range(8 * kMemoryWidthBytes - 1,
                       8 * (kMemoryWidthBytes - bytesMissing)) =
          read.range(8 * bytesMissing - 1, 0);
      MemoryPack_t next;
      next.data().range(8 * (sizeof(T) - bytesMissing) - 1, 0) =
          read.range(8 * sizeof(T) - 1, 8 * bytesMissing);
      streamOut.WriteBlocking(out);
      out = next;
      bytesMissing = 64 - sizeof(T) + bytesMissing;
    } else {  // bytesMissing > sizeof(T)
      out.data().range(8 * (kMemoryWidthBytes - bytesMissing + sizeof(T)) - 1,
                       8 * (kMemoryWidthBytes - bytesMissing)) = read;
      bytesMissing = bytesMissing - sizeof(T);
    }
  }
}
