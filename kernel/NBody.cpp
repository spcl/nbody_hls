#include "NBody.h"
#include "hlslib/Simulation.h"
#include "hlslib/Stream.h"
#include "hlslib/DataPack.h"

// CURRENT STATUS: need to flatten the loop for the saturation stage :-/

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

void ProcessingElement(hlslib::Stream<Packed> &streamIn,
                       hlslib::Stream<Packed> &streamOut, int d) {
  Vec_t pos, nextPos;

Time:
  for (int t = 0; t < kSteps; ++t) {

  Outer:
    for (int n = 0; n < kN / kTileSize + 1; ++n) {
    Inner:
      for (int m = 0; m < kN; ++m) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        if (n == 0) {
          const auto read = hlslib::ReadBlocking(streamIn);
          if (n == d) {
            pos = read.pos;
          }
          if (n < kTileSize - d - 1) {
            hlslib::WriteBlocking(streamOut, read);
          }
        } else { // n > 0 
          const auto in = hlslib::ReadBlocking(streamIn);
          if (n == m) {
            // Don't compute force with self 
            hlslib::WriteBlocking(streamOut, in);
          } else {
            Vec_t acc;
          Dims:
            for (int i = 0; i < kDims; ++i) {
              #pragma HLS UNROLL
              acc[i] =
                  in.acc[i] + ComputeAcceleration(in.mass, pos[i], in.pos[i]);
            }
            const Packed out(acc, in.pos, in.mass); 
            if (m == n + kTileSize) {
              // Store for when we process next tile 
              nextPos = in.pos;
            }
            hlslib::WriteBlocking(streamOut, out);
          }
        } // End n > 0
      } // End loop M
    } // End loop N

  } // End loop T
}

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

void ReadMemory(Data_t const memory[],
                hlslib::Stream<Data_t> &stream) {
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

void PackData(hlslib::Stream<Vec_t> &posIn,
              hlslib::Stream<Data_t> &massIn,
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

void NBody(Data_t const mass[], Vec_t const positionIn[], Vec_t positionOut[],
           Vec_t const velocityIn[], Vec_t velocityOut[]) {

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
  hlslib::Stream<Data_t> massPipe("massPipe");
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
  HLSLIB_DATAFLOW_FUNCTION(ReadMemory, mass, massPipe);
  HLSLIB_DATAFLOW_FUNCTION(PackData, posPipeIn, massPipe, packedPipes[0]);

  for (int nt = 0; nt < kTileSize; ++nt) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, packedPipes[nt],
                             packedPipes[nt + 1], nt);
  }

  HLSLIB_DATAFLOW_FUNCTION(UpdateVelocityAndPosition, packedPipes[kTileSize],
                           velPipeIn, velPipeOut, posPipeOut);
  HLSLIB_DATAFLOW_FUNCTION(WritePosition, posPipeOut, positionOut);
  HLSLIB_DATAFLOW_FUNCTION(WriteVelocity, velPipeOut, velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
           
}
