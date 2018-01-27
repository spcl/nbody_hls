#include "NBody.h"
#include "hls_stream.h"
#include "hlslib/DataPack.h"
#include "hlslib/Simulation.h"
#include "hlslib/Stream.h"

// Notes:
// SoA seems like a good idea, but switching between dimensions becomes a pain,
// because we will need to load the base elements twice. We also have to load
// mass for every dimension.
// We would only store pos_x and the next pos_x'.
//
// AoS is messy because of the wide fan-out of operations if we combine it with
// SIMD. Perhaps this is okay if done carefully?
// The advantage is that we only need to stream mass through once per all
// dimensions, instead of per dimension. Additionally, we avoid the mess of
// switching between dimensions, which doesn't pipeline well.
//
// Initially a pull architecture seemed good, since we would only have to read
// and write velocity once per timestep. Unfortunately this would leave us with
// the accumulation problem: we would be trying to accumulate every cycle in
// the same variable.
// Instead we try a push-based architecture, where each PE updates the input
// acceleration with its own contribution. This means we have to read and write
// velocities every iteration.

void ProcessingElementFlattened(hlslib::Stream<Packed> &streamIn,
                                hlslib::Stream<Packed> &streamOut, int depth) {
  Vec_t pos, nextPos;

Time:
  for (int t = 0; t < kSteps; ++t) {
    int n = 0;
    int m = 0;
  Outer:
    for (int i = 0; i < (kN / kTileSize) * kN + kTileSize - depth; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1

      const auto in = hlslib::ReadBlocking(streamIn);

      if (i < kTileSize - depth) {
        if (i == 0) {
          nextPos = in.pos;
        } else {
          hlslib::WriteBlocking(streamOut, in);
        }

      } else {  // n >= kTileSize - depth

        if (m == 0) {
          pos = nextPos;
        }

        // Don't compute force with self
        if (n != m) {
          const auto acc = in.acc + ComputeAcceleration(in.mass, pos, in.pos);
          const Packed out(acc, in.pos, in.mass);
          if (m == n + kTileSize) {
            // Store for when we process next tile
            nextPos = in.pos;
          }
          hlslib::WriteBlocking(streamOut, out);
        } else {
          hlslib::WriteBlocking(streamOut, in);
        }

        // Update flattened indices
        if (m == kN - 1) {
          m = 0;
          if (n == kN / kTileSize - 1) {
            n = 0;
          } else {
            ++n;
          }
        } else {
          ++m;
        }

      }  // End else

    }  // End flattened loop

  }  // End loop T
}

// void ProcessingElementSimple(hlslib::Stream<Packed> &streamIn,
//                              hlslib::Stream<Packed> &streamOut, int d) {
//   Vec_t pos, nextPos;
//
// Time:
//   for (int t = 0; t < kSteps; ++t) {
//
//     for (int m = 0; m < kTileSize - d; ++m) {
//       #pragma HLS PIPELINE II=1
//       const auto read = hlslib::ReadBlocking(streamIn);
//       if (m == 0) {
//         nextPos = read.pos;
//       } else {
//         hlslib::WriteBlocking(streamOut, read);
//       }
//     }
//
//   Outer:
//     for (int n = 0; n < kN / kTileSize; ++n) {
//     Inner:
//       for (int m = 0; m < kN; ++m) {
//         #pragma HLS LOOP_FLATTEN
//         #pragma HLS PIPELINE II=1
//         if (m == 0) {
//           pos = nextPos;
//         }
//         const auto in = hlslib::ReadBlocking(streamIn);
//         if (n == m) {
//           // Don't compute force with self
//           hlslib::WriteBlocking(streamOut, in);
//         } else {
//           Vec_t acc;
//         Dims:
//           for (int i = 0; i < kDims; ++i) {
//             #pragma HLS UNROLL
//             acc[i] =
//                 in.acc[i] + ComputeAcceleration(in.mass, pos[i], in.pos[i]);
//           }
//           const Packed out(acc, in.pos, in.mass);
//           if (m == n + kTileSize) {
//             // Store for when we process next tile
//             nextPos = in.pos;
//           }
//           hlslib::WriteBlocking(streamOut, out);
//         }
//       } // End loop M
//     } // End loop N
//
//   } // End loop T
// }

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

void RepeatFirstTile(hlslib::Stream<Packed> &streamIn,
                     hlslib::Stream<Packed> &streamOut) {
  hls::stream<Packed> buffer;
  #pragma HLS STREAM variable=buffer depth=kTileSize
Time:
  for (int t = 0; t < kSteps; ++t) {
  Flattened:
    for (int i = 0; i < kN / kTileSize * kN + kTileSize; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      Packed out;
      if (i < kN / kTileSize * kN) {
        out = hlslib::ReadBlocking(streamIn);
      } else {
        out = buffer.read();
      }
      if (i < kTileSize) {
        buffer.write(out);
      }
      hlslib::WriteBlocking(streamOut, out);
    }
  }
}

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

void ReadMass(MemoryPack_t const memory[],
              hlslib::Stream<MemoryPack_t> &stream) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Outer:
    for (int n = 0; n < kN / kTileSize; ++n) {
    Inner:
      for (int m = 0; m < hlslib::CeilDivide(kN, kMemoryWidth); ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        hlslib::WriteBlocking(stream, memory[m]);
      }
    }
  }
}

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

template <unsigned iterations, typename T>
void ConvertMemoryWidth(hlslib::Stream<MemoryPack_t> &streamIn,
                        hlslib::Stream<T> &streamOut) {
  // Takes a stream of wide memory accesses and converts it into elements of a
  // size that do not divide into the memory width. When crossing a memory
  // boundary, use bytes from both the previous and next memory access.
  ap_uint<6> bytesRemaining = 0;
  MemoryPack_t curr;
  MemoryPack_t next;
  ap_uint<8 * sizeof(T)> out;
  for (unsigned i = 0; i < iterations; ++i) {
    #pragma HLS PIPELINE II=1
    if (bytesRemaining < sizeof(T)) {
      next = hlslib::ReadBlocking(streamIn);
      out.range(8 * sizeof(T) - 1, 8 * (sizeof(T) - bytesRemaining)) =
          curr.data().range(8 * bytesRemaining - 1, 0);
      out.range(8 * (sizeof(T) - bytesRemaining) - 1, 0) = next.data().range(
          8 * kMemoryWidthBytes - 1,
          8 * (kMemoryWidthBytes - (sizeof(T) - bytesRemaining)));
      bytesRemaining = kMemoryWidthBytes - (sizeof(T) - bytesRemaining);
      curr = next;
    } else {
      out = curr.data().range(8 * bytesRemaining - 1,
                              8 * (bytesRemaining - sizeof(T)));
      bytesRemaining = bytesRemaining - sizeof(T);
    }
    hlslib::WriteBlocking(streamOut, out);
  }
}

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

template <typename T, unsigned iterations>
void PackMemory(hlslib::Stream<T> &streamIn,
                hlslib::Stream<MemoryPack_t> &streamOut) {
  constexpr int kElementsPerMemory = sizeof(MemoryPack_t) / sizeof(T);
  static_assert(sizeof(MemoryPack_t) % sizeof(T) == 0, "Must be divisible");
  for (unsigned i = 0; i < iterations / kElementsPerMemory; ++i) {
    MemoryPack_t mem;
    for (unsigned j = 0; j < kElementsPerMemory; ++j) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      mem[j] = hlslib::ReadBlocking(streamIn);
      if (j == kElementsPerMemory - 1) {
        hlslib::WriteBlocking(streamOut, mem);
      }
    }
  }
}

void NBody(MemoryPack_t const mass[], Vec_t const positionIn[],
           Vec_t positionOut[], Vec_t const velocityIn[], Vec_t velocityOut[]) {
           
  #pragma HLS INTERFACE m_axi port=mass offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=positionIn offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=positionOut offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=velocityIn offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi port=velocityOut offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=mass bundle=control
  #pragma HLS INTERFACE s_axilite port=positionIn bundle=control
  #pragma HLS INTERFACE s_axilite port=positionOut bundle=control
  #pragma HLS INTERFACE s_axilite port=velocityIn bundle=control
  #pragma HLS INTERFACE s_axilite port=velocityOut bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  hlslib::Stream<Vec_t> posPipeIn("posPipeIn");
  hlslib::Stream<Vec_t> posPipeOut("posPipeOut");
  hlslib::Stream<Vec_t> velPipeIn("velPipeIn");
  hlslib::Stream<Vec_t> velPipeOut("velPipeOut");
  hlslib::Stream<MemoryPack_t> massMemPipe("massMemPipe");
  hlslib::Stream<Data_t> massPipe("massPipe");
  hlslib::Stream<Packed> repeatPipe("posPipeInRepeat");
  hlslib::Stream<Packed> packedPipes[kTileSize + 1];
  #pragma HLS DATA_PACK variable=packedPipes
#ifndef NBODY_SYNTHESIS
  for (int i = 0; i < kTileSize + 1; ++i) {
    packedPipes[i].set_name("packedPipes[" + std::to_string(i) + "]");
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadVector, positionIn, posPipeIn);
  HLSLIB_DATAFLOW_FUNCTION(ReadVector, velocityIn, velPipeIn);
  HLSLIB_DATAFLOW_FUNCTION(ReadMass, mass, massMemPipe);
  HLSLIB_DATAFLOW_FUNCTION(UnpackMass, massMemPipe, massPipe);
  HLSLIB_DATAFLOW_FUNCTION(PackData, posPipeIn, massPipe, repeatPipe);
  HLSLIB_DATAFLOW_FUNCTION(RepeatFirstTile, repeatPipe, packedPipes[0]);

  for (int nt = 0; nt < kTileSize; ++nt) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElementFlattened, packedPipes[nt],
                             packedPipes[nt + 1], nt);
  }

  HLSLIB_DATAFLOW_FUNCTION(UpdateVelocityAndPosition, packedPipes[kTileSize],
                           velPipeIn, velPipeOut, posPipeOut);
  HLSLIB_DATAFLOW_FUNCTION(WritePosition, posPipeOut, positionOut);
  HLSLIB_DATAFLOW_FUNCTION(WriteVelocity, velPipeOut, velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
}
