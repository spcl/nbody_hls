#include "NBody.h"
#include "Memory.h"
#include "hls_stream.h"
#include "hlslib/DataPack.h"
#include "hlslib/Simulation.h"
#include "hlslib/Stream.h"

// Notes(Simon): The CUDA implementation assumes that the input is given by vectors of the
// form (x, y, z, m), (better access pattern, might want to assume that as well?).
// Cuda also alters the computation a little to avoid having values go to infinity, they claim
// this is done in practice: f(i,j) = G*m(i)*m(j)*r(i,j)/(scalar_prod(r(i,j)) + eps^2)^(3/2)
// (Essentially this adds a softening factor epsilon to avoid dividing by 0, motivated by
// galaxies actually going through each other instead of colliding. This also means that the
// force of an object on itself does not need a extra case distinction as it will safely
// evaluate to 0.)

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
// void ProcessingElements(hlslib::Stream<PosMass_t> &posMassIn,
//                         hlslib::Stream<PosMass_t> &posMassOut,
//                         hlslib::Stream<Vec_t> &velocityIn,
//                         hlslib::Stream<Vec_t> &velocityOut) {
//   for (int t = 0; t < kSteps; ++t) {
//
//     PosMass_t posWeightBuffer[kUnrollDepth][2*kPipelineFactor];
//     #pragma HLS ARRAY_PARTITION variable=posWeightBuffer dim=1 complete
//
//     Vec_t acc[kUnrollDepth][kPipelineFactor];
//     #pragma HLS ARRAY_PARTITION variable=acc dim=1 complete
//
//     int next = 0;
//
//     // Buffer first tile
//   BufferFirstTile:
//     for (int k = 0; k < kUnrollDepth; ++k) {
//       for (int l = 0; l < kPipelineFactor; ++l) {
//         #pragma HLS PIPELINE II=1
//         posWeightBuffer[k][l] = posMassIn.Pop();
//         Vec_t a(static_cast<Data_t>(0));
//         acc[k][l] = a;
//       }
//     }
//     // Loop over tiles
//   ComputeTiles:
//     for (int bn = 0; bn < kNTiles; ++bn) {
//       next = 1 - next;
//     ComputeBodies:
//       for (int i = 0; i < kNBodies; ++i) {
//         #pragma HLS PIPELINE II=1
//         PosMass_t s1 = posMassIn.Pop();
//         if (i >= (bn + 1) * kUnrollDepth * kPipelineFactor &&
//             i < (bn + 2) * kUnrollDepth * kPipelineFactor &&
//             bn != kNBodies / (kUnrollDepth * kPipelineFactor) - 1) {
//           int a = i - (bn + 1) * kUnrollDepth * kPipelineFactor;
//           posWeightBuffer[a / kPipelineFactor]
//                          [a % kPipelineFactor + (next ? kPipelineFactor : 0)] =
//                              s1;
//         }
//
//       ComputePipeline:
//         for (int l = 0; l < kPipelineFactor; ++l) {
//           #pragma HLS PIPELINE II=1
//         ComputeUnroll:
//           for (int k = 0; k < kUnrollDepth; ++k) {
//             #pragma HLS UNROLL
//             PosMass_t s0 = posWeightBuffer[k][l + (next ? 0 : kPipelineFactor)];
//             #pragma HLS DEPENDENCE variable=posWeightBuffer inter false
//
//             Vec_t tmpacc = ComputeAcceleration<true>(s0, s1);
//             acc[k][l] = acc[k][l] + tmpacc;
//             #pragma HLS DEPENDENCE variable=acc inter false
//           }
//         }
//       }
//       // Write out result
//     WriteUnroll:
//       for (int k = 0; k < kUnrollDepth; ++k) {
//       WritePipeline:
//         for (int l = 0; l < kPipelineFactor; ++l) {
//           #pragma HLS PIPELINE II=1
//           Vec_t vel = velocityIn.Pop();
//           PosMass_t pm;
//           pm[kDims] =
//               posWeightBuffer[k][l + (next ? 0 : kPipelineFactor)][kDims];
//         WriteDims:
//           for (int s = 0; s < kDims; s++) {
//             #pragma HLS UNROLL
//             pm[s] = posWeightBuffer[k][l + (next ? 0 : kPipelineFactor)][s];
//             vel[s] = vel[s] + acc[k][l][s]*kTimestep;
//             pm[s] = pm[s] + vel[s]*kTimestep;
//           }
//           posMassOut.Push(pm);
//           velocityOut.Push(vel);
//           Vec_t a(static_cast<Data_t>(0));
//           acc[k][l] = a;
//         }
//       }
//     }
//
//   }
// }

void DummyKernel(hlslib::Stream<PosMass_t> &posMassIn,
                 hlslib::Stream<PosMass_t> &posMassOut,
                 hlslib::Stream<Vec_t> &velocityIn,
                 hlslib::Stream<Vec_t> &velocityOut) {
  for (int t = 0; t < kSteps; ++t) {

  BufferUnroll:
    for (int k = 0; k < kUnrollDepth; ++k) {
    BufferPipeline:
      for (int l = 0; l < kPipelineFactor; ++l) {
        #pragma HLS PIPELINE II=1
        posMassIn.Pop();
      }
    }

  ComputeTiles:
    for (int bn = 0; bn < kNTiles; ++bn) {
    ComputeBodies:
      for (int i = 0; i < kNBodies; ++i) {
        #pragma HLS PIPELINE II=1
        posMassIn.Pop();
      ComputePipeline:
        for (int l = 0; l < kPipelineFactor; ++l) {
          #pragma HLS PIPELINE II=1
        ComputeUnroll:
          for (int k = 0; k < kUnrollDepth; ++k) {
            #pragma HLS UNROLL
          }
        }
      }
      // Write out result
    WriteUnroll:
      for (int k = 0; k < kUnrollDepth; ++k) {
      WritePipeline:
        for (int l = 0; l < kPipelineFactor; ++l) {
          #pragma HLS PIPELINE II=1
          velocityOut.Push(velocityIn.Pop());
          posMassOut.Push(PosMass_t(5.));
        }
      }
    }

  }
}

void NBody(MemoryPack_t const positionMassIn[], MemoryPack_t positionMassOut[],
           MemoryPack_t const velocityIn[], MemoryPack_t velocityOut[],
           hlslib::Stream<MemoryPack_t> &velocityReadMemory,
           hlslib::Stream<Vec_t> &velocityReadKernel,
           hlslib::Stream<Vec_t> &velocityWriteKernel,
           hlslib::Stream<MemoryPack_t> &velocityWriteMemory) {

  #pragma HLS INTERFACE m_axi port=positionMassIn offset=slave  bundle=gmem0
  #pragma HLS INTERFACE m_axi port=positionMassOut offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=velocityIn offset=slave      bundle=gmem1
  #pragma HLS INTERFACE m_axi port=velocityOut offset=slave     bundle=gmem1
  #pragma HLS INTERFACE axis port=velocityReadMemory
  #pragma HLS INTERFACE axis port=velocityReadKernel
  #pragma HLS INTERFACE axis port=velocityWriteMemory
  #pragma HLS INTERFACE axis port=velocityWriteKernel
  #pragma HLS INTERFACE s_axilite port=positionMassIn  bundle=control
  #pragma HLS INTERFACE s_axilite port=positionMassOut bundle=control
  #pragma HLS INTERFACE s_axilite port=velocityIn      bundle=control
  #pragma HLS INTERFACE s_axilite port=velocityOut     bundle=control
  #pragma HLS INTERFACE s_axilite port=return          bundle=control

  #pragma HLS DATAFLOW

  hlslib::Stream<MemoryPack_t> posMassInMemoryPipe("posMassInMemoryPipe");
  hlslib::Stream<MemoryPack_t> posMassOutMemoryPipe("posMassOutMemoryPipe");
  hlslib::Stream<PosMass_t> posMassInPipe("posMassInPipe");
  hlslib::Stream<PosMass_t> posMassInRepeatPipe("posMassInRepeatPipe");
  hlslib::Stream<PosMass_t> posMassOutPipe("posMassOutPipe");

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadMemory_PositionMass, positionMassIn,
                           posMassInMemoryPipe);

  HLSLIB_DATAFLOW_FUNCTION(ContractWidth_PositionMass, posMassInMemoryPipe,
                           posMassInRepeatPipe);

  HLSLIB_DATAFLOW_FUNCTION(RepeatFirstTile, posMassInRepeatPipe, posMassInPipe);

  HLSLIB_DATAFLOW_FUNCTION(ReadMemory_Velocity, velocityIn, velocityReadMemory);

#ifndef HLSLIB_SYNTHESIS
  ::hlslib::_Dataflow::Get().AddFunction(
      ConvertMemoryToNonDivisible<Data_t, kDims>, velocityReadMemory,
      velocityReadKernel, kSteps * kNBodies);
  ::hlslib::_Dataflow::Get().AddFunction(
      ConvertNonDivisibleToMemory<Data_t, kDims>, velocityWriteKernel,
      velocityWriteMemory, kSteps * kNBodies);
#endif

  HLSLIB_DATAFLOW_FUNCTION(DummyKernel, posMassInPipe, posMassOutPipe,
                           velocityReadKernel, velocityWriteKernel);

  HLSLIB_DATAFLOW_FUNCTION(ExpandWidth_PositionMass, posMassOutPipe,
                           posMassOutMemoryPipe);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_PositionMass, posMassOutMemoryPipe,
                           positionMassOut);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_Velocity, velocityWriteMemory,
                           velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
}
