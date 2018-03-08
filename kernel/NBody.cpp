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
void Compute(hlslib::Stream<PosMass_t> &posMassIn,
             hlslib::Stream<PosMass_t> &posMassOut,
             hlslib::Stream<Vec_t> &velocityIn,
             hlslib::Stream<Vec_t> &velocityOut) {
  for (int t = 0; t < kSteps; ++t) {
    PosMass_t posWeightBuffer[2][kUnrollDepth][kPipelineFactor];
    Vec_t acc[kUnrollDepth][kPipelineFactor];
    int next = 0;

    // Buffer first tile
    for (int k = 0; k < kUnrollDepth; ++k) {
      for (int l = 0; l < kPipelineFactor; ++l) {
        posWeightBuffer[0][k][l] = posMassIn.Pop();
        Vec_t a(static_cast<Data_t>(0));
        acc[k][l] = a;
      }
    }
    // Loop over tiles
    for (int bn = 0; bn < kNTiles; ++bn) {
      next = 1 - next;
      for (int i = 0; i < kNBodies; ++i) {
        #pragma HLS PIPELINE II=1
        PosMass_t s1 = posMassIn.Pop();
        if (i >= (bn + 1) * kUnrollDepth * kPipelineFactor &&
            i < (bn + 2) * kUnrollDepth * kPipelineFactor &&
            bn != kNBodies / (kUnrollDepth * kPipelineFactor) - 1) {
          int a = i - (bn + 1) * kUnrollDepth * kPipelineFactor;
          posWeightBuffer[next][a / kPipelineFactor][a % kPipelineFactor] = s1;
        }

        for (int l = 0; l < kPipelineFactor; ++l) {
          #pragma HLS PIPELINE II=1
          for (int k = 0; k < kUnrollDepth; ++k) {
            #pragma HLS UNROLL
            PosMass_t s0 = posWeightBuffer[1 - next][k][l];

            Vec_t tmpacc = ComputeAcceleration<true>(s0, s1);
            acc[k][l] = acc[k][l] + tmpacc;
          }
        }
      }
      // Write out result
      for (int k = 0; k < kUnrollDepth; ++k) {
        for (int l = 0; l < kPipelineFactor; ++l) {
          Vec_t vel = velocityIn.Pop();
          PosMass_t pm;
          pm[kDims] = posWeightBuffer[1-next][k][l][kDims];
          for (int s = 0; s < kDims; s++) {
            pm[s] = posWeightBuffer[1-next][k][l][s];
            vel[s] = vel[s] + acc[k][l][s]*kTimestep;
            pm[s] = pm[s] + vel[s]*kTimestep;
          }
          posMassOut.Push(pm);
          velocityOut.Push(vel);
          Vec_t a(static_cast<Data_t>(0));
          acc[k][l] = a;
        }
      }
    }

  }
}

void NBody(MemoryPack_t const positionMassIn[], MemoryPack_t positionMassOut[],
           Vec_t const velocityIn[], Vec_t velocityOut[]) {

  #pragma HLS INTERFACE m_axi port=positionMassIn offset=slave  bundle=gmem0
  #pragma HLS INTERFACE m_axi port=positionMassOut offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=velocityIn offset=slave      bundle=gmem1
  #pragma HLS INTERFACE m_axi port=velocityOut offset=slave     bundle=gmem1
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
  hlslib::Stream<Vec_t> velocityInPipe("velocityInPipe");
  hlslib::Stream<Vec_t> velocityOutPipe("velocityOutPipe");

  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadMemory_PositionMass, positionMassIn,
                           posMassInMemoryPipe);

  HLSLIB_DATAFLOW_FUNCTION(ContractWidth_PositionMass, posMassInMemoryPipe,
                           posMassInRepeatPipe);

  HLSLIB_DATAFLOW_FUNCTION(RepeatFirstTile, posMassInRepeatPipe, posMassInPipe);

  HLSLIB_DATAFLOW_FUNCTION(ReadSingle_Velocity, velocityIn, velocityInPipe);

  HLSLIB_DATAFLOW_FUNCTION(Compute, posMassInPipe, posMassOutPipe,
                           velocityInPipe, velocityOutPipe);

  HLSLIB_DATAFLOW_FUNCTION(ExpandWidth_PositionMass, posMassOutPipe,
                           posMassOutMemoryPipe);

  HLSLIB_DATAFLOW_FUNCTION(WriteMemory_PositionMass, posMassOutMemoryPipe,
                           positionMassOut);

  HLSLIB_DATAFLOW_FUNCTION(WriteSingle_Velocity, velocityOutPipe, velocityOut);

  HLSLIB_DATAFLOW_FINALIZE();
}
