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

void ComputeSumUpEnd(hlslib::Stream<PosMass_t> &posMassIn,
             hlslib::Stream<PosMass_t> &posMassOut,
             hlslib::Stream<Vec_t> &velocityIn,
             hlslib::Stream<Vec_t> &velocityOut,
             hlslib::Stream<Vec_t> &accIn,
             hlslib::Stream<Vec_t> &accOut,
             int d) {
Time:
  for (int t = 0; t < kSteps; ++t) {

    PosMass_t posWeightBuffer[2 * kPipelineFactor];
    Vec_t acc[kPipelineFactor];

    bool next = false;


  SaturateBuffer:
    for (int i = 0; i < (kUnrollDepth - d) * kPipelineFactor; i++) {
      #pragma HLS PIPELINE II=1
      PosMass_t pm = posMassIn.Pop();
      if (i < (kUnrollDepth - 1 - d) * kPipelineFactor) {
        posMassOut.Push(pm);
      } else {
        posWeightBuffer[i % kPipelineFactor] = pm;
        Vec_t a(static_cast<Data_t>(0));
        acc[i % kPipelineFactor] = a;
      }
    }

  ComputeTiles:
    for (int bn = 0; bn < kNTiles; bn++) {

    ComputeBodies:
      for (int j = 0; j < kNBodies; j++) {

        PosMass_t s1;

      ComputePipeline:
        for (int l = 0; l < kPipelineFactor; ++l) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN

          if (j == 0 && l == 0) {
            next = !next;
          }

          // First loop iteration
          if (l == 0) {
            s1 = posMassIn.Pop();
            if (d != kUnrollDepth - 1) {
              posMassOut.Push(s1);
            }

            if (j >= (bn + 1) * kUnrollDepth * kPipelineFactor &&
                j < (bn + 2) * kUnrollDepth * kPipelineFactor &&
                bn != kNBodies / (kUnrollDepth * kPipelineFactor) - 1) {
              int a = j - (bn + 1) * kUnrollDepth * kPipelineFactor;
              if (a / kPipelineFactor == kUnrollDepth - d - 1) {
                posWeightBuffer[a % kPipelineFactor +
                                (next ? kPipelineFactor : 0)] = s1;
              }
            }
          }
          // End first loop iteration

          // All iterations (actual compute)
          PosMass_t s0 = posWeightBuffer[l + (next ? 0 : kPipelineFactor)];
          #pragma HLS DEPENDENCE variable=posWeightBuffer inter false
          // TODO: is this dependence pragma safe? Can we use FIFOs?
          Vec_t tmpacc = ComputeAcceleration<true>(s0, s1);
          acc[l] = acc[l] + tmpacc;
          #pragma HLS DEPENDENCE variable=acc inter false
        }
      }

    WriteDepth:
      for (int i = 0; i < kUnrollDepth; i++) {
      WritePipeline:
        for (int j = 0; j < kPipelineFactor; j++) {
          #pragma HLS PIPELINE II=1
          if(d == kUnrollDepth - 1){
            //Compute
            if(i == 0){
              Vec_t vel = velocityIn.Pop();
              PosMass_t pm;
              pm[kDims] =
                  posWeightBuffer[j + (next ? 0 : kPipelineFactor)][kDims];

            WriteDimsFirst:
              for (int s = 0; s < kDims; s++) {
                #pragma HLS UNROLL
                pm[s] = posWeightBuffer[j + (next ? 0 : kPipelineFactor)][s];
                vel[s] = vel[s] + acc[j][s] * kTimestep;
                pm[s] = pm[s] + vel[s] * kTimestep;
              }
              posMassOut.Push(pm);
              velocityOut.Push(vel);
              Vec_t a(static_cast<Data_t>(0));
              acc[j] = a;
            }else{
              Vec_t vel = velocityIn.Pop();
              PosMass_t pm = posMassIn.Pop();
              Vec_t accP = accIn.Pop();

            WriteDimsOther:
              for (int s = 0; s < kDims; s++) {
                #pragma HLS UNROLL
                vel[s] = vel[s] + accP[s] * kTimestep;
                pm[s] = pm[s] + vel[s] * kTimestep;
              }
              posMassOut.Push(pm);
              velocityOut.Push(vel);
            }

          }else{
            if (i > kUnrollDepth - d - 1) {
              PosMass_t pm = posMassIn.Pop();
              Vec_t vel = velocityIn.Pop();
              Vec_t accP = accIn.Pop();

              posMassOut.Push(pm);
              velocityOut.Push(vel);
              accOut.Push(accP);
            } else if (i == kUnrollDepth - d - 1) {
              Vec_t vel = velocityIn.Pop();
              PosMass_t pm;
              pm = posWeightBuffer[j + (next ? 0 : kPipelineFactor)];
              posMassOut.Push(pm);
              velocityOut.Push(vel);
              accOut.Push(acc[j]);
              Vec_t a(static_cast<Data_t>(0));
              acc[j] = a;
            } else {
              Vec_t vel = velocityIn.Pop();
              velocityOut.Push(vel);
            }
            }
          }
        }
      }
    }
}

void Compute(hlslib::Stream<PosMass_t> &posMassIn,
             hlslib::Stream<PosMass_t> &posMassOut,
             hlslib::Stream<Vec_t> &velocityIn,
             hlslib::Stream<Vec_t> &velocityOut, int d) {
Time:
  for (int t = 0; t < kSteps; ++t) {

    PosMass_t posWeightBuffer[2 * kPipelineFactor];
    Vec_t acc[kPipelineFactor];

    bool next = false;

  SaturateBuffer:
    for (int i = 0; i < (kUnrollDepth - d) * kPipelineFactor; i++) {
      #pragma HLS PIPELINE II=1
      PosMass_t pm = posMassIn.Pop();
      if (i < (kUnrollDepth - 1 - d) * kPipelineFactor) {
        posMassOut.Push(pm);
      } else {
        posWeightBuffer[i % kPipelineFactor] = pm;
        Vec_t a(static_cast<Data_t>(0));
        acc[i % kPipelineFactor] = a;
      }
    }

  ComputeTiles:
    for (int bn = 0; bn < kNTiles; bn++) {

    ComputeBodies:
      for (int j = 0; j < kNBodies; j++) {

        PosMass_t s1;

      ComputePipeline:
        for (int l = 0; l < kPipelineFactor; ++l) {
          #pragma HLS PIPELINE II=1
          #pragma HLS LOOP_FLATTEN

          if (j == 0 && l == 0) {
            next = !next;
          }

          // First loop iteration
          if (l == 0) {
            s1 = posMassIn.Pop();
            if (d != kUnrollDepth - 1) {
              posMassOut.Push(s1);
            }

            if (j >= (bn + 1) * kUnrollDepth * kPipelineFactor &&
                j < (bn + 2) * kUnrollDepth * kPipelineFactor &&
                bn != kNBodies / (kUnrollDepth * kPipelineFactor) - 1) {
              int a = j - (bn + 1) * kUnrollDepth * kPipelineFactor;
              if (a / kPipelineFactor == kUnrollDepth - d - 1) {
                posWeightBuffer[a % kPipelineFactor +
                                (next ? kPipelineFactor : 0)] = s1;
              }
            }
          }
          // End first loop iteration

          // All iterations (actual compute)
          PosMass_t s0 = posWeightBuffer[l + (next ? 0 : kPipelineFactor)];
          #pragma HLS DEPENDENCE variable=posWeightBuffer inter false
          // TODO: is this dependence pragma safe? Can we use FIFOs?
          Vec_t tmpacc = ComputeAcceleration<true>(s0, s1);
          acc[l] = acc[l] + tmpacc;
          #pragma HLS DEPENDENCE variable=acc inter false
        }
      }

    WriteDepth:
      for (int i = 0; i < kUnrollDepth; i++) {
      WritePipeline:
        for (int j = 0; j < kPipelineFactor; j++) {
          #pragma HLS PIPELINE II=1
          if (i > kUnrollDepth - d - 1) {
            PosMass_t pm = posMassIn.Pop();
            Vec_t vel = velocityIn.Pop();
            posMassOut.Push(pm);
            velocityOut.Push(vel);
          } else if (i == kUnrollDepth - d - 1) {
            // Compute
            Vec_t vel = velocityIn.Pop();
            PosMass_t pm;
            pm[kDims] =
                posWeightBuffer[j + (next ? 0 : kPipelineFactor)][kDims];

          WriteDims:
            for (int s = 0; s < kDims; s++) {
              #pragma HLS UNROLL
              pm[s] = posWeightBuffer[j + (next ? 0 : kPipelineFactor)][s];
              vel[s] = vel[s] + acc[j][s] * kTimestep;
              pm[s] = pm[s] + vel[s] * kTimestep;
            }

            posMassOut.Push(pm);
            velocityOut.Push(vel);
            Vec_t a(static_cast<Data_t>(0));
            acc[j] = a;
          } else {
            Vec_t vel = velocityIn.Pop();
            PosMass_t s1;
            velocityOut.Push(vel);
          }
        }
      }
    }
  }
}

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

  hlslib::Stream<PosMass_t> pm_pipes[kUnrollDepth - 1];
  hlslib::Stream<Vec_t> vel_pipes[kUnrollDepth - 1];
  hlslib::Stream<Vec_t> acc_pipes[kUnrollDepth - 1];

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

  HLSLIB_DATAFLOW_FUNCTION(ComputeSumUpEnd, posMassInPipe, pm_pipes[0],
                           velocityReadKernel, vel_pipes[0], acc_pipes[0], acc_pipes[0], 0);

  for (int i = 1; i < kUnrollDepth - 1; i++) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ComputeSumUpEnd, pm_pipes[i - 1], pm_pipes[i],
                             vel_pipes[i - 1], vel_pipes[i], acc_pipes[i - 1], acc_pipes[i], i);
  }

  HLSLIB_DATAFLOW_FUNCTION(ComputeSumUpEnd, pm_pipes[kUnrollDepth - 2], posMassOutPipe,
                           vel_pipes[kUnrollDepth - 2], velocityWriteKernel, acc_pipes[kUnrollDepth - 2], acc_pipes[kUnrollDepth - 2],
                           kUnrollDepth - 1);

  HLSLIB_DATAFLOW_FUNCTION(ExpandWidth_PositionMass, posMassOutPipe,
                           posMassOutMemoryPipe);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_PositionMass, posMassOutMemoryPipe,
                           positionMassOut);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_Velocity, velocityWriteMemory,
                           velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
}
