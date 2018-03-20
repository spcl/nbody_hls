/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "Memory.h"

// Tiling scheme:
//
// T := kSteps
// N := kNBodies
// K := kUnrollDepth
// L := kPipelineFactor
//
// for t in 0:T
//   for k in 0:K
//     for l in 0:L
//       load(i)
//   for i in N/(K*L)
//     for j in N
//       load(j)
//       for l in L
//         parallel_for k in K
//           compute
//   for k in 0:K
//     for l in 0:L
//       store(i)

void ReadMemory_PositionMass(MemoryPack_t const memory[],
                             hlslib::Stream<MemoryPack_t> &pipe) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Tiles:
    for (int bn = 0; bn < kNTiles; ++bn) {
    Domain:
      for (int i = 0; i < kMemorySizePosition; ++i) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE II=1
        const auto index = (t % 2 == 0) ? i : (kMemorySizePosition + i);
        pipe.Push(memory[index]);
      }
    }
  }
}

void ContractWidth_PositionMass(hlslib::Stream<MemoryPack_t> &wide,
                                hlslib::Stream<PosMass_t> &narrow) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Tiles:
    for (int bn = 0; bn < kNTiles; ++bn) {
    Memory:
      for (int i = 0; i < kMemorySizePosition; ++i) {
        MemoryPack_t mem;
      Kernel:
        for (int j = 0; j < kVectorsPerMemory; ++j) {
          #pragma HLS LOOP_FLATTEN
          #pragma HLS PIPELINE II=1
          if (j == 0) {
            mem = wide.Pop();
          }
          PosMass_t posMass;
        VectorCopy:
          for (int k = 0; k < kDims + 1; ++k) {
            #pragma HLS UNROLL
            posMass[k] = mem[j * (kDims + 1) + k];
          }
          narrow.Push(posMass);
        }
      }
    }
  }
}

void ExpandWidth_PositionMass(hlslib::Stream<PosMass_t> &narrow,
                              hlslib::Stream<MemoryPack_t> &wide) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Memory:
    for (int i = 0; i < kMemorySizePosition; ++i) {
      MemoryPack_t mem;
    Kernel:
      for (int j = 0; j < kVectorsPerMemory; ++j) {
        const auto read = narrow.Pop();
        for (int k = 0; k < kDims + 1; ++k) {
          #pragma HLS UNROLL
          mem[j * (kDims + 1) + k] = read[k];
        }
        if (j == kVectorsPerMemory - 1) {
          wide.Push(mem);
        }
      }
    }
  }
}

void WriteMemory_PositionMass(hlslib::Stream<MemoryPack_t> &pipe,
                              MemoryPack_t memory[]) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Domain:
    for (int i = 0; i < kMemorySizePosition; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      const auto index = (t % 2 == 0) ? (kMemorySizePosition + i) : i;
      memory[index] = pipe.Pop();
    }
  }
}

void RepeatFirstTile(hlslib::Stream<PosMass_t> &streamIn,
                     hlslib::Stream<PosMass_t> &streamOut) {
  hlslib::Stream<PosMass_t> buffer(kUnrollDepth * kPipelineFactor);
Time:
  for (int t = 0; t < kSteps; ++t) {
  TilesMemoryFlattened:
    for (int i = 0; i < kNTiles * kNBodies + (kUnrollDepth * kPipelineFactor);
         ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      PosMass_t value;
      if ((i < kUnrollDepth * kPipelineFactor) ||
          (i >= 2 * kUnrollDepth * kPipelineFactor)) {
        value = streamIn.Pop();
      } else {
        value = buffer.Pop();
      }
      if (i < kUnrollDepth * kPipelineFactor) {
        buffer.Push(value);
      }
      streamOut.Push(value);
    }
  }
}

void ReadMemory_Velocity(MemoryPack_t const memory[],
                         hlslib::Stream<MemoryPack_t> &pipe) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Domain:
    for (int i = 0; i < kMemorySizeVelocity; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      pipe.Push(memory[i]);
    }
  }
}

// // Used for testing software. Does not work with AXI if kDims is 3
// void ReadSingle_Velocity(Vec_t const memory[],
//                          hlslib::Stream<Vec_t> &pipe) {
// Time:
//   for (int t = 0; t < kSteps; ++t) {
//   Domain:
//     for (int i = 0; i < kNBodies; ++i) {
//       #pragma HLS LOOP_FLATTEN
//       #pragma HLS PIPELINE II=1
//       pipe.Push(memory[i]);
//     }
//   }
// }

void WriteMemory_Velocity(hlslib::Stream<MemoryPack_t> &pipe,
                          MemoryPack_t memory[]) {
Time:
  for (int t = 0; t < kSteps; ++t) {
  Domain:
    for (int i = 0; i < kMemorySizeVelocity; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      memory[i] = pipe.Pop();
    }
  }
}

// // Used for testing software. Does not work with AXI if kDims is 3
// void WriteSingle_Velocity(hlslib::Stream<Vec_t> &pipe,
//                           Vec_t memory[]) {
// Time:
//   for (int t = 0; t < kSteps; ++t) {
//   Domain:
//     for (int i = 0; i < kNBodies; ++i) {
//       #pragma HLS LOOP_FLATTEN
//       #pragma HLS PIPELINE II=1
//       memory[i] = pipe.Pop();
//     }
//   }
// }
