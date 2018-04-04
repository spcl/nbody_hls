#include "NBody.h"
#include "Memory.h"
#include "hlslib/DataPack.h"
#include "hlslib/Simulation.h"
#include "hlslib/Stream.h"
#include <cassert>

// Notes(Simon): The CUDA implementation assumes that the input is given by
// vectors of the form (x, y, z, m), (better access pattern, might want to
// assume that as well?). Cuda also alters the computation a little to avoid
// having values go to infinity, they claim this is done in practice: f(i,j) =
// G*m(i)*m(j)*r(i,j)/(scalar_prod(r(i,j)) + eps^2)^(3/2) (Essentially this adds
// a softening factor epsilon to avoid dividing by 0, motivated by galaxies
// actually going through each other instead of colliding. This also means that
// the force of an object on itself does not need a extra case distinction as it
// will safely evaluate to 0.)

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

#ifndef NBODY_FLATTEN

// void ComputeSumUpEnd(hlslib::Stream<PosMass_t> &posMassIn,
//                      hlslib::Stream<PosMass_t> &posMassOut,
//                      hlslib::Stream<Vec_t> &velocityIn,
//                      hlslib::Stream<Vec_t> &velocityOut,
//                      hlslib::Stream<Vec_t> &accIn,
//                      hlslib::Stream<Vec_t> &accOut, int d) {
// Time:
//   for (int t = 0; t < kSteps; ++t) {
//     PosMass_t posWeightBuffer[2 * kPipelineFactor];
//     Vec_t acc[kPipelineFactor];
//
//     bool next = false;
//
//   SaturateBuffer:
//     for (int i = 0; i < (kUnrollDepth - d) * kPipelineFactor; i++) {
//       #pragma HLS PIPELINE II=1
//       PosMass_t pm = posMassIn.Pop();
//       if (i < (kUnrollDepth - 1 - d) * kPipelineFactor) {
//         posMassOut.Push(pm);
//       } else {
//         posWeightBuffer[i % kPipelineFactor] = pm;
//         Vec_t a(static_cast<Data_t>(0));
//         acc[i % kPipelineFactor] = a;
//       }
//     }
//
//   ComputeTiles:
//     for (int bn = 0; bn < kNTiles; bn++) {
//     ComputeBodies:
//       for (int j = 0; j < kNBodies; j++) {
//         PosMass_t s1;
//
//       ComputePipeline:
//         for (int l = 0; l < kPipelineFactor; ++l) {
//           #pragma HLS PIPELINE II=1
//           #pragma HLS LOOP_FLATTEN
//
//           if (j == 0 && l == 0) {
//             next = !next;
//           }
//
//           // First loop iteration
//           if (l == 0) {
//             s1 = posMassIn.Pop();
//             if (d != kUnrollDepth - 1) {
//               posMassOut.Push(s1);
//             }
//
//             if (j >= (bn + 1) * kUnrollDepth * kPipelineFactor &&
//                 j < (bn + 2) * kUnrollDepth * kPipelineFactor &&
//                 bn != kNBodies / (kUnrollDepth * kPipelineFactor) - 1) {
//               int a = j - (bn + 1) * kUnrollDepth * kPipelineFactor;
//               if (a / kPipelineFactor == kUnrollDepth - d - 1) {
//                 posWeightBuffer[a % kPipelineFactor +
//                                 (next ? kPipelineFactor : 0)] = s1;
//               }
//             }
//           }
//           // End first loop iteration
//
//           // All iterations (actual compute)
//           PosMass_t s0 = posWeightBuffer[l + (next ? 0 : kPipelineFactor)];
//           #pragma HLS DEPENDENCE variable=posWeightBuffer inter false
//           // TODO: is this dependence pragma safe? Can we use FIFOs?
//           Vec_t tmpacc = ComputeAcceleration<true>(s0, s1);
//           acc[l] = acc[l] + tmpacc;
//           #pragma HLS DEPENDENCE variable=acc inter false
//         }
//       }
//
//     WriteDepth:
//       for (int i = 0; i < kUnrollDepth; i++) {
//       WritePipeline:
//         for (int j = 0; j < kPipelineFactor; j++) {
//           #pragma HLS PIPELINE II=1
//           if (d == kUnrollDepth - 1) {
//             // Compute
//             if (i == 0) {
//               Vec_t vel = velocityIn.Pop();
//               PosMass_t pm;
//               pm[kDims] =
//                   posWeightBuffer[j + (next ? 0 : kPipelineFactor)][kDims];
//
//             WriteDimsFirst:
//               for (int s = 0; s < kDims; s++) {
//                 #pragma HLS UNROLL
//                 pm[s] = posWeightBuffer[j + (next ? 0 : kPipelineFactor)][s];
//                 vel[s] = vel[s] + acc[j][s] * kTimestep;
//                 pm[s] = pm[s] + vel[s] * kTimestep;
//               }
//               posMassOut.Push(pm);
//               velocityOut.Push(vel);
//               Vec_t a(static_cast<Data_t>(0));
//               acc[j] = a;
//             } else {
//               Vec_t vel = velocityIn.Pop();
//               PosMass_t pm = posMassIn.Pop();
//               Vec_t accP = accIn.Pop();
//
//             WriteDimsOther:
//               for (int s = 0; s < kDims; s++) {
//                 #pragma HLS UNROLL
//                 vel[s] = vel[s] + accP[s] * kTimestep;
//                 pm[s] = pm[s] + vel[s] * kTimestep;
//               }
//               posMassOut.Push(pm);
//               velocityOut.Push(vel);
//             }
//
//           } else {
//             if (i > kUnrollDepth - d - 1) {
//               PosMass_t pm = posMassIn.Pop();
//               Vec_t vel = velocityIn.Pop();
//               Vec_t accP = accIn.Pop();
//
//               posMassOut.Push(pm);
//               velocityOut.Push(vel);
//               accOut.Push(accP);
//             } else if (i == kUnrollDepth - d - 1) {
//               Vec_t vel = velocityIn.Pop();
//               PosMass_t pm;
//               pm = posWeightBuffer[j + (next ? 0 : kPipelineFactor)];
//               posMassOut.Push(pm);
//               velocityOut.Push(vel);
//               accOut.Push(acc[j]);
//               Vec_t a(static_cast<Data_t>(0));
//               acc[j] = a;
//             } else {
//               Vec_t vel = velocityIn.Pop();
//               velocityOut.Push(vel);
//             }
//           }
//         }
//       }
//     }
//   }
// }

void ComputeStage(hlslib::Stream<PosMass_t> &posMassIn,
                  hlslib::Stream<PosMass_t> &posMassOut,
                  hlslib::Stream<Vec_t> &velocityIn,
                  hlslib::Stream<Vec_t> &velocityOut,
                  hlslib::Stream<Vec_t> &accelerationIn,
                  hlslib::Stream<Vec_t> &accelerationOut, int d) {
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

            // We currently hold values [bn*(K*L) + d*L, (bn+1)*(K*L) + d*L]
            // We need values [bn*(K*L) + (d+1)*L, (bn+1)*(K*L) + (d+1)*L]
            if (j >= ((bn + 1) * kUnrollDepth * kPipelineFactor +
                      (kUnrollDepth - 1 - d) * kPipelineFactor) &&
                j < ((bn + 1) * kUnrollDepth * kPipelineFactor +
                     (kUnrollDepth - d) * kPipelineFactor)) {
              posWeightBuffer[j % kPipelineFactor +
                              (next ? kPipelineFactor : 0)] = s1;
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
          if (i < kUnrollDepth - d - 1) {
            // Until own index is reached, forward velocities to their
            // respective processing elements
            velocityOut.Push(velocityIn.Pop());
          } else if (i == kUnrollDepth - d - 1) {
            // Once at our own index, push the final computed acceleration
            // along with the positions and velocities
            const Vec_t vel = velocityIn.Pop();
            const PosMass_t pm =
                posWeightBuffer[j + (next ? 0 : kPipelineFactor)];
            posMassOut.Push(pm);
            velocityOut.Push(vel);
            accelerationOut.Push(acc[j]);
            // Reset accumulation variables
            acc[j] = Vec_t(static_cast<Data_t>(0));
          } else {
            // For the remaining iterations, just forward the packaged/
            // positions, velocities and accelerations of previous processing
            // elements
            const auto pm = posMassIn.Pop();
            const auto vel = velocityIn.Pop();
            posMassOut.Push(pm);
            velocityOut.Push(vel);
            if (d != 0) {
              // HLS cannot figure out that this never happens, so put an
              // explicit "assertion" here
              const auto accRead = accelerationIn.Pop();
              accelerationOut.Push(accRead);
            }
          }
        }
      }
    }
  }
}

#else // NBODY_FLATTEN

void ComputeStage(hlslib::Stream<PosMass_t> &posMassIn,
                  hlslib::Stream<PosMass_t> &posMassOut,
                  hlslib::Stream<Vec_t> &velocityIn,
                  hlslib::Stream<Vec_t> &velocityOut,
                  hlslib::Stream<Vec_t> &accelerationIn,
                  hlslib::Stream<Vec_t> &accelerationOut,
                  int d) {

  ap_uint<hlslib::ConstLog2(kUnrollDepth * kPipelineFactor)> s = 0;
  ap_uint<hlslib::ConstLog2(kNBodies)> j = 0;
  ap_uint<hlslib::ConstLog2(kPipelineFactor)> l0 = 0;
  ap_uint<hlslib::ConstLog2(kPipelineFactor)> l1 = 0;
  ap_uint<hlslib::ConstLog2(kUnrollDepth)> k = 0;
  ap_uint<hlslib::ConstLog2(kNTiles)> bn = 0;

  bool next = false;

  enum class State {
    saturating,
    streaming,
    draining
  };
  State state = State::saturating;

Time:
  for (int t = 0; t < kSteps; ++t) {
    PosMass_t posWeightBuffer[2 * kPipelineFactor];
    Vec_t acc[kPipelineFactor];

    PosMass_t s1;

  Flattened:
    for (int _i = 0; _i < (kUnrollDepth - d) * kPipelineFactor +
                              kNTiles * ((kNBodies * kPipelineFactor) +
                                         (kUnrollDepth * kPipelineFactor));
         ++_i) {
      if (state == State::saturating) {
        // --------------------------------------------------------------------

        PosMass_t pm = posMassIn.Pop();
        if (s < (kUnrollDepth - 1 - d) * kPipelineFactor) {
          posMassOut.Push(pm);
        } else {
          posWeightBuffer[s % kPipelineFactor] = pm;
          Vec_t a(static_cast<Data_t>(0));
          acc[s % kPipelineFactor] = a;
        }

        if (s == (kUnrollDepth - d) * kPipelineFactor - 1) {
          s = 0;
          state = State::streaming;
        } else {
          ++s;
        }

        // --------------------------------------------------------------------
      } else if (state == State::streaming) {
        // --------------------------------------------------------------------

        if (j == 0 && l0 == 0) {
          next = !next;
        }

        // First loop iteration
        if (l0 == 0) {
          s1 = posMassIn.Pop();
          if (d != kUnrollDepth - 1) {
            posMassOut.Push(s1);
          }

          // We currently hold values [bn*(K*L) + d*L, (bn+1)*(K*L) + d*L]
          // We need values [bn*(K*L) + (d+1)*L, (bn+1)*(K*L) + (d+1)*L]
          if (j >= ((bn + 1) * kUnrollDepth * kPipelineFactor +
                    (kUnrollDepth - 1 - d) * kPipelineFactor) &&
              j < ((bn + 1) * kUnrollDepth * kPipelineFactor +
                   (kUnrollDepth - d) * kPipelineFactor)) {
            posWeightBuffer[j % kPipelineFactor +
                            (next ? kPipelineFactor : 0)] = s1;
          }

        }
        // End first loop iteration

        // All iterations (actual compute)
        PosMass_t s0 = posWeightBuffer[l0 + (next ? 0 : kPipelineFactor)];
        #pragma HLS DEPENDENCE variable=posWeightBuffer inter false
        // TODO: is this dependence pragma safe? Can we use FIFOs?
        Vec_t tmpacc = ComputeAcceleration<true>(s0, s1);
        acc[l0] = acc[l0] + tmpacc;
        #pragma HLS DEPENDENCE variable=acc inter false

        if (l0 == kPipelineFactor - 1) {
          l0 = 0;
          if (j == kNBodies - 1) {
            j = 0;
            state = State::draining;
          } else {
            ++j;
          }
        } else {
          ++l0;
        }

        // --------------------------------------------------------------------
      } else { // state == State::draining 
        // --------------------------------------------------------------------

        if (k < kUnrollDepth - d - 1) {
          // Until own index is reached, forward velocities to their
          // respective processing elements
          velocityOut.Push(velocityIn.Pop());
        } else if (k == kUnrollDepth - d - 1) {
          // Once at our own index, push the final computed acceleration
          // along with the positions and velocities
          const Vec_t vel = velocityIn.Pop();
          const PosMass_t pm =
              posWeightBuffer[l1 + (next ? 0 : kPipelineFactor)];
          posMassOut.Push(pm);
          velocityOut.Push(vel);
          accelerationOut.Push(acc[l1]);
          // Reset accumulation variables
          acc[l1] = Vec_t(static_cast<Data_t>(0));
        } else {
          // For the remaining iterations, just forward the packaged/
          // positions, velocities and accelerations of previous processing
          // elements
          const auto pm = posMassIn.Pop();
          const auto vel = velocityIn.Pop();
          posMassOut.Push(pm);
          velocityOut.Push(vel);
          if (d != 0) {
            // HLS cannot figure out that this never happens, so put an
            // explicit "assertion" here
            const auto accRead = accelerationIn.Pop();
            accelerationOut.Push(accRead);
          }
        }

        if (l1 == kPipelineFactor - 1) {
          l1 = 0;
          if (k == kUnrollDepth - 1) {
            k = 0;
            if (bn == kNTiles - 1) {
              bn = 0;
              state = State::saturating;
            } else {
              ++bn;
              state = State::streaming;
            }
          } else {
            ++k;
          }
        } else {
          ++l1;
        }

        // --------------------------------------------------------------------
      } // State == state::draining

    } // Flattened loop 
  } // Loop over steps
}

#endif // NBODY_FLATTEN

void UpdateBodies(hlslib::Stream<Vec_t> &accelerationIn,
                  hlslib::Stream<Vec_t> &velocityIn,
                  hlslib::Stream<Vec_t> &velocityOut,
                  hlslib::Stream<PosMass_t> &positionMassIn,
                  hlslib::Stream<PosMass_t> &positionMassOut) {
Update_Steps:
  for (int t = 0; t < kSteps; ++t) {
  Update_N:
    for (int i = 0; i < kNBodies; ++i) {
      #pragma HLS LOOP_FLATTEN
      #pragma HLS PIPELINE II=1
      const auto acc = accelerationIn.Pop();
      const auto vel = velocityIn.Pop();
      const auto pm = positionMassIn.Pop();
      Vec_t velNew;
      PosMass_t pmNew;
    Update_Dims:
      for (int d = 0; d < kDims; d++) {
        #pragma HLS UNROLL
        velNew[d] = vel[d] + acc[d] * kTimestep;
        pmNew[d] = pm[d] + velNew[d] * kTimestep;
      }
      pmNew[kDims] = pm[kDims];
      velocityOut.Push(velNew);
      positionMassOut.Push(pmNew);
    }
  }
}

void NBody(MemoryPack_t const positionMassIn[], MemoryPack_t positionMassOut[],
           MemoryPack_t const velocityIn[], MemoryPack_t velocityOut[],
           // 512-bit to 96-bit memory width conversion
           hlslib::Stream<MemoryPack_t> &velocityReadMemory,
           hlslib::Stream<Vec_t> &velocityReadKernel,
           // 512-bit to 128-bit memory width conversion
           hlslib::Stream<MemoryPack_t> &positionMassReadMemory,
           hlslib::Stream<PosMass_t> &positionMassReadKernel,
           // 96-bit to 512-bit memory width conversion
           hlslib::Stream<Vec_t> &velocityWriteKernel,
           hlslib::Stream<MemoryPack_t> &velocityWriteMemory,
           // 128-bit to 512-bit memory width conversion
           hlslib::Stream<PosMass_t> &positionMassWriteKernel,
           hlslib::Stream<MemoryPack_t> &positionMassWriteMemory) {

  #pragma HLS INTERFACE m_axi port=positionMassIn offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=positionMassOut offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=velocityIn offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=velocityOut offset=slave bundle=gmem1
  #pragma HLS INTERFACE axis port=velocityReadMemory
  #pragma HLS INTERFACE axis port=velocityReadKernel
  #pragma HLS INTERFACE axis port=positionMassReadMemory
  #pragma HLS INTERFACE axis port=positionMassReadKernel
  #pragma HLS INTERFACE axis port=velocityWriteMemory
  #pragma HLS INTERFACE axis port=velocityWriteKernel
  #pragma HLS INTERFACE axis port=positionMassWriteMemory
  #pragma HLS INTERFACE axis port=positionMassWriteKernel
  #pragma HLS INTERFACE s_axilite port=positionMassIn bundle=control
  #pragma HLS INTERFACE s_axilite port=positionMassOut bundle=control
  #pragma HLS INTERFACE s_axilite port=velocityIn bundle=control
  #pragma HLS INTERFACE s_axilite port=velocityOut bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  
  #pragma HLS DATAFLOW

  hlslib::Stream<PosMass_t> positionMassRepeat("positionMassRepeat");

  hlslib::Stream<PosMass_t> positionMassPipes[kUnrollDepth];
  hlslib::Stream<Vec_t> velocityPipes[kUnrollDepth];
  hlslib::Stream<Vec_t> accelerationPipes[kUnrollDepth + 1];

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadMemory_PositionMass, positionMassIn,
                           positionMassReadMemory);

  HLSLIB_DATAFLOW_FUNCTION(RepeatFirstTile, positionMassReadKernel,
                           positionMassRepeat);

  HLSLIB_DATAFLOW_FUNCTION(ReadMemory_Velocity, velocityIn, velocityReadMemory);

#ifndef HLSLIB_SYNTHESIS
  for (int i = 0; i < kUnrollDepth; ++i) {
    positionMassPipes[i].set_name(
        ("positionMassPipes[" + std::to_string(i) + "]").c_str());
    velocityPipes[i].set_name(
        ("velocityPipes[" + std::to_string(i) + "]").c_str());
    accelerationPipes[i].set_name(
        ("accelerationPipes[" + std::to_string(i) + "]").c_str());
  }
  accelerationPipes[kUnrollDepth].set_name(
      ("accelerationPipes[" + std::to_string(kUnrollDepth) + "]").c_str());
  ::hlslib::_Dataflow::Get().AddFunction(
      ConvertMemoryToNonDivisible<Data_t, kDims>, velocityReadMemory,
      velocityReadKernel, kSteps * kNBodies);
  ::hlslib::_Dataflow::Get().AddFunction(
      ConvertNonDivisibleToMemory<Data_t, kDims>, velocityWriteKernel,
      velocityWriteMemory, kSteps * kNBodies);
  HLSLIB_DATAFLOW_FUNCTION(ContractWidth_PositionMass, positionMassReadMemory,
                           positionMassReadKernel);
  HLSLIB_DATAFLOW_FUNCTION(ExpandWidth_PositionMass, positionMassWriteKernel,
                           positionMassWriteMemory);
#endif

  HLSLIB_DATAFLOW_FUNCTION(ComputeStage, positionMassRepeat,
                           positionMassPipes[0], velocityReadKernel,
                           velocityPipes[0], accelerationPipes[0],
                           accelerationPipes[1], 0);

  for (int i = 1; i < kUnrollDepth; i++) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ComputeStage, positionMassPipes[i - 1],
                             positionMassPipes[i], velocityPipes[i - 1],
                             velocityPipes[i], accelerationPipes[i],
                             accelerationPipes[i + 1], i);
  }

  HLSLIB_DATAFLOW_FUNCTION(
      UpdateBodies,
      accelerationPipes[kUnrollDepth],
      velocityPipes[kUnrollDepth - 1],
      velocityWriteKernel,
      positionMassPipes[kUnrollDepth - 1],
      positionMassWriteKernel);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_PositionMass, positionMassWriteMemory,
                           positionMassOut);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_Velocity, velocityWriteMemory,
                           velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
}
