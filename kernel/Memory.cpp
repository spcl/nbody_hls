/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "Memory.h"

/// Receives velocities from memory and updates them with acceleration values
/// received from the kernel, and computes the updated positions when a
/// timestep finishes.
void UpdateVelocityAndPosition(hlslib::Stream<Packed> &fromKernel,
                               hlslib::Stream<Vec_t> &velocityIn,
                               hlslib::Stream<Vec_t> &velocityOut,
                               hlslib::Stream<Vec_t> &positionOut) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kN; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        const auto v = hlslib::ReadBlocking(velocityIn);
        const auto pack = hlslib::ReadBlocking(fromKernel);
        Vec_t vNew;
      DimsVelocity:
        for (int d = 0; d < kDims; ++d) {
          #pragma HLS UNROLL
          vNew[d] = v[d] + pack.acc[d] * kTimestep;
        }
        hlslib::WriteBlocking(velocityOut, vNew);
        if (n == kN / kTileSize - 1) {
          Vec_t sNew;
        DimsPosition:
          for (int d = 0; d < kDims; ++d) {
            #pragma HLS UNROLL
            sNew[d] = pack.pos[d] + vNew[d] * kTimestep;
          }
          hlslib::WriteBlocking(positionOut, sNew);
        }
      }
    }
  }
}

/// Directly read vectors from memory. Does not work if the vector is not of
/// binary size
void ReadVector(Vec_t const memory[], hlslib::Stream<Vec_t> &stream) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kN; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        hlslib::WriteBlocking(stream, memory[m]);
      }
    }
  }
}

/// Reads 512-bit values from memory, to be converted into vectors by a
/// subsequent module
void ReadVectorMemory(MemoryPack_t const memory[],
                      hlslib::Stream<MemoryPack_t> &stream) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kMemoryPerTile; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        stream.WriteBlocking(memory[m]);
      }
    }
  }
}

/// Every timestep, buffer and repeat the first tile, as this must be stored in
/// the processing elements before streaming can begin
void RepeatFirstTile(hlslib::Stream<Packed> &streamIn,
                     hlslib::Stream<Packed> &streamOut) {
  hlslib::Stream<Packed> buffer("repeatBuffer", kTileSize);
Time:
  for (int t = 0; t < kSteps; ++t) {
  Flattened:
    for (int i = 0; i < kN / kTileSize * kN + kTileSize; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      Packed out;
      if (i < kN / kTileSize * kN) {
        out = streamIn.ReadBlocking();
      } else {
        out = buffer.ReadBlocking();
      }
      if (i < kTileSize) {
        buffer.write(out);
      }
      streamOut.WriteBlocking(out);
    }
  }
}

/// Directly writes velocity vectors to memory. Does not work if the vector is
/// not of binary size
void WriteVelocity(hlslib::Stream<Vec_t> &stream, Vec_t memory[]) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kN; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        memory[m] = hlslib::ReadBlocking(stream);
      }
    }
  }
}

/// Directly writes position vectors to memory. Does not work if the vector is
/// not of binary size
void WritePosition(hlslib::Stream<Vec_t> &stream, Vec_t memory[]) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN; ++n) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      memory[n] = hlslib::ReadBlocking(stream);
    }
  }
}

/// Writes velocity vectors to memory after they've been converted to the memory
/// width.
void WriteVelocityMemory(hlslib::Stream<MemoryPack_t> &stream,
                         MemoryPack_t memory[]) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kMemoryPerTile; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        memory[m] = hlslib::ReadBlocking(stream);
      }
    }
  }
}

/// Directly writes position vectors to memory. Does not work if the vector is
/// not of binary size
void WritePositionMemory(hlslib::Stream<MemoryPack_t> &stream,
                         MemoryPack_t memory[]) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kMemoryPerTile; ++n) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      memory[n] = hlslib::ReadBlocking(stream);
    }
  }
}

/// Reads 512-bit wide vectors of mass values to be converted into scalars or
/// shorter vectors by the subsequent module
void ReadMass(MemoryPack_t const memory[],
              hlslib::Stream<MemoryPack_t> &stream) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kN / kMemoryWidth; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        hlslib::WriteBlocking(stream, memory[m]);
      }
    }
  }
}

/// Packs separate position and mass streams into a single packed bus to be
/// passed to the kernel
void PackData(hlslib::Stream<Vec_t> &posIn, hlslib::Stream<Data_t> &massIn,
              hlslib::Stream<Packed> &packedOut) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < kN; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        const Packed packed(Vec_t{static_cast<Data_t>(0)},
                            hlslib::ReadBlocking(posIn),
                            hlslib::ReadBlocking(massIn));
        hlslib::WriteBlocking(packedOut, packed);
      }
    }
  }
}

/// Narrows 512-bit wide vectors from memory into scalar mass elements
void UnpackMass(hlslib::Stream<MemoryPack_t> &streamIn,
                hlslib::Stream<Data_t> &streamOut) {
  #pragma HLS INLINE
  constexpr int kElementsPerMemory = sizeof(MemoryPack_t) / sizeof(Data_t);
  static_assert(sizeof(MemoryPack_t) % sizeof(Data_t) == 0,
                "Must be divisible");
  MemoryPack_t mem;
  ap_uint<hlslib::ConstLog2(kElementsPerMemory)> i_mem = 0;
  for (unsigned t = 0; t < kSteps; ++t) {
    for (unsigned n = 0; n < kN / kTileSize; ++n) {
      for (unsigned m = 0; m < kN; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        if (i_mem == 0) {
          mem = hlslib::ReadBlocking(streamIn);
        }
        const Data_t part = mem[i_mem];
        hlslib::WriteBlocking(streamOut, part);
        if (i_mem == kElementsPerMemory - 1 || m == kN - 1) {
          i_mem = 0;
        } else {
          ++i_mem;
        }
      }
    }
  }
}

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
void ConvertMemoryToVector(hlslib::Stream<MemoryPack_t> &streamIn,
                           hlslib::Stream<Vec_t> &streamOut) {
  #pragma HLS INLINE
  ConvertMemoryToNonDivisible<kSteps *(kN / kTileSize) * kN, Vec_t>(streamIn,
                                                                    streamOut);
}

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
void ConvertVelocityToMemory(hlslib::Stream<Vec_t> &streamIn,
                             hlslib::Stream<MemoryPack_t> &streamOut) {
  #pragma HLS INLINE
  ConvertNonDivisibleToMemory<kSteps *(kN / kTileSize) * kN, Vec_t>(streamIn,
                                                                    streamOut);
}

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
void ConvertPositionToMemory(hlslib::Stream<Vec_t> &streamIn,
                             hlslib::Stream<MemoryPack_t> &streamOut) {
  #pragma HLS INLINE
  ConvertNonDivisibleToMemory<kSteps * kN, Vec_t>(streamIn, streamOut);
}
