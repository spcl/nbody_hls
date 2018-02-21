#include "NBody.h"
#include "Memory.h"
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
// This in turn means considerably higher load on the memory bandwidth, but as
// long as we stay under the bandwidth limit this should come at no hit to
// performance.

// Notes(Simon): The CUDA implementation assumes that the input is given by vectors of the
// form (x, y, z, m), (better access pattern, might want to assume that as well?).
// Cuda also alters the computation a little to avoid having values go to infinity, they claim
// this is done in practice: f(i,j) = G*m(i)*m(j)*r(i,j)/(scalar_prod(r(i,j)) + eps^2)^(3/2)
// (Essentially this adds a softening factor epsilon to avoid dividing by 0, motivated by
// galaxies actually going through each other instead of colliding. This also means that the
// force of an object on itself does not need a extra case distinction as it will safely
// evaluate to 0.)

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

void NBody(MemoryPack_t const mass[], MemoryPack_t const positionIn[],
           MemoryPack_t positionOut[], MemoryPack_t const velocityIn[],
           MemoryPack_t velocityOut[]) {
           
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

  hlslib::Stream<MemoryPack_t> posPipeInMemory("posPipeInMemory");
  hlslib::Stream<Vec_t> posPipeIn("posPipeIn");
  hlslib::Stream<Vec_t> posPipeOut("posPipeOut");
  hlslib::Stream<MemoryPack_t> posPipeOutMemory("posPipeOutMemory");
  hlslib::Stream<MemoryPack_t> velPipeInMemory("velPipeInMemory");
  hlslib::Stream<Vec_t> velPipeIn("velPipeIn");
  hlslib::Stream<Vec_t> velPipeOut("velPipeOut");
  hlslib::Stream<MemoryPack_t> velPipeOutMemory("velPipeOutMemory");
  hlslib::Stream<MemoryPack_t> massPipeMemory("massPipeMemory");
  hlslib::Stream<Data_t> massPipe("massPipe");
  hlslib::Stream<Packed> repeatPipe("repeatPipe");
  hlslib::Stream<Packed> packedPipes[kTileSize + 1];
  #pragma HLS DATA_PACK variable=packedPipes
#ifndef NBODY_SYNTHESIS
  for (int i = 0; i < kTileSize + 1; ++i) {
    packedPipes[i].set_name("packedPipes[" + std::to_string(i) + "]");
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadVectorMemory, positionIn, posPipeInMemory);
  HLSLIB_DATAFLOW_FUNCTION(ConvertMemoryToVector, posPipeInMemory, posPipeIn);
  HLSLIB_DATAFLOW_FUNCTION(ReadVectorMemory, velocityIn, velPipeInMemory);
  HLSLIB_DATAFLOW_FUNCTION(ConvertMemoryToVector, velPipeInMemory, velPipeIn);
  HLSLIB_DATAFLOW_FUNCTION(ReadMass, mass, massPipeMemory);
  HLSLIB_DATAFLOW_FUNCTION(UnpackMass, massPipeMemory, massPipe);
  HLSLIB_DATAFLOW_FUNCTION(PackData, posPipeIn, massPipe, repeatPipe);
  HLSLIB_DATAFLOW_FUNCTION(RepeatFirstTile, repeatPipe, packedPipes[0]);

  for (int nt = 0; nt < kTileSize; ++nt) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElementFlattened, packedPipes[nt],
                             packedPipes[nt + 1], nt);
  }

  HLSLIB_DATAFLOW_FUNCTION(UpdateVelocityAndPosition, packedPipes[kTileSize],
                           velPipeIn, velPipeOut, posPipeOut);
  HLSLIB_DATAFLOW_FUNCTION(ConvertPositionToMemory, posPipeOut,
                           posPipeOutMemory);
  HLSLIB_DATAFLOW_FUNCTION(ConvertVelocityToMemory, velPipeOut,
                           velPipeOutMemory);
  HLSLIB_DATAFLOW_FUNCTION(WritePositionMemory, posPipeOutMemory, positionOut);
  HLSLIB_DATAFLOW_FUNCTION(WriteVelocityMemory, velPipeOutMemory, velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
}
