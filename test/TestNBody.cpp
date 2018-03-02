/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "NBody.h"
#include "Utility.h"
#include <cmath>

void NewAlgorithmReference(hlslib::DataPack<Data_t, kDims + 1> positionMass[],
                           Vec_t velocity[]);

int main() {
  std::vector<Data_t> mass(kN);
  std::vector<Vec_t> velocity(kN);
  std::vector<Vec_t> positionRef(kN);
  std::vector<hlslib::DataPack<Data_t, kDims + 1>> posWeight(kN);

  //maybe make those configurable in the future :), if we use them
  float clusterScale = 1.54f;
  float velocityScale = 8.0f;

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> distMass(1e10, 1e12);
  std::uniform_real_distribution<double> distVelocity(-1e3, 1e3);
  std::uniform_real_distribution<double> distPosition(-1e6, 1e6);

  // std::for_each(mass.begin(), mass.end(),
  //               [&distMass, &rng](Data_t &in) { in = distMass(rng); });
  // std::for_each(velocity.begin(), velocity.end(),
  //               [&distVelocity, &rng](Vec_t &in) {
  //                 for (int d = 0; d < kDims; ++d) {
  //                   in[d] = distVelocity(rng);
  //                 }
  //               });
  // std::for_each(position.begin(), position.end(),
  //               [&distPosition, &rng](Vec_t &in) {
  //                 for (int d = 0; d < kDims; ++d) {
  //                   in[d] = distPosition(rng);
  //                 }
  //               });

  //our random initialisation doesn't really work, so I got the one from CUDA
  float scale = clusterScale * std::max<Data_t>(1.0, kN/(1024.0));
  float vscale = velocityScale * scale;

  int i = 0;

  while(i < kN){
    Vec_t point;
    float lenSqr = 0.0;
    for(int j = 0; j < kDims; j++){
      point[j] = rand()/(float) RAND_MAX * 2 - 1;
      lenSqr += point[j]*point[j];
    }
    if(lenSqr > 1) continue;

    Vec_t vel;
    lenSqr = 0.0;
    for(int j = 0; j < kDims; j++){
      vel[j] = rand()/(float) RAND_MAX * 2 - 1;
      lenSqr += vel[j]*vel[j];
    }
    if(lenSqr > 1) continue;

    for(int j = 0; j < kDims; j++){
      posWeight[i][j] = point[j]*scale;
      positionRef[i][j] = point[j]*scale;
      velocity[i][j] = vel[j]*vscale;
    }

    posWeight[i][kDims] = 1.0f; //mass
    mass[i] = 1.0f; //massForReference

    i++;
  }

  std::vector<Vec_t> velocityRef(velocity);
  std::vector<Vec_t> position(positionRef);


  // NBody(reinterpret_cast<MemoryPack_t const *>(&mass[0]),
  //       reinterpret_cast<MemoryPack_t const *>(&position[0]),
  //       reinterpret_cast<MemoryPack_t *>(&position[0]),
  //       reinterpret_cast<MemoryPack_t const *>(&velocity[0]),
  //       reinterpret_cast<MemoryPack_t *>(&velocity[0]));

  NewAlgorithmReference(posWeight.data(), velocity.data());
  // Reference(mass.data(), position.data(), velocity.data());
  ReferenceLikeCUDA(mass.data(), positionRef.data(), velocityRef.data());

  for (unsigned i = 0; i < std::min<unsigned>(kN, 100); ++i) {
    std::cout << posWeight[i] << " / " << positionRef[i] << ", " << velocity[i]
              << " / " << velocityRef[i] << "\n";
  }

  return 0;
}

// Reference implementation for new hardware design
void NewAlgorithmReference(hlslib::DataPack<Data_t, kDims + 1> positionMass[],
                           Vec_t velocity[]) {
  for (int t = 0; t < kSteps; t++) {
    hlslib::DataPack<Data_t, kDims + 1> positionMassNew[kN];
    Vec_t velocityNew[kN];
    // is a datapack initialised to zero by default?
    hlslib::DataPack<Data_t, kDims + 1> posWeight[2][kTileSize]
                                                 [kDepthProcessingElement];
    Vec_t acc[kTileSize][kDepthProcessingElement];
    int next = 0;

    for (int i = 0; i < kTileSize; i++) {
      for (int j = 0; j < kDepthProcessingElement; j++) {
        for (int k = 0; k < kDims + 1; k++) {
          posWeight[0][i][j][k] =
              positionMass[i * kDepthProcessingElement + j][k];
          if (k != kDims) acc[i][j][k] = 0.0;
        }
      }
    }
    for (int i = 0; i < kN / (kDepthProcessingElement * kTileSize); i++) {
      next = 1 - next;
      for (int j = 0; j < kN; j++) {
        hlslib::DataPack<Data_t, kDims + 1> currentPos;
        for (int k = 0; k < kDims + 1; k++) {
          currentPos[k] = positionMass[j][k];

          // Here I populate the next buffer with appropriate elments.
          if (j >= (i + 1) * kDepthProcessingElement * kTileSize &&
              j < (i + 2) * kDepthProcessingElement * kTileSize &&
              i != kN / (kDepthProcessingElement * kTileSize) - 1) {
            int a = j - (i + 1) * kDepthProcessingElement * kTileSize;
            posWeight[next][a / kDepthProcessingElement]
                     [a % kDepthProcessingElement][k] = currentPos[k];
          }
        }
        // Now comes the second unroll, this time per processing element
        for (int l = 0; l < kDepthProcessingElement; l++) {
          // The loop that is replicated in Hardware
          for (int k = 0; k < kTileSize; k++) {
            Vec_t s0;
            Vec_t s1;

            // Accounts for the fact that ComputeAccelerationSoftened does not
            // take posWeight args yet
            for (int s = 0; s < kDims; s++) {
              s0[s] = posWeight[1 - next][k][l][s];
              s1[s] = currentPos[s];
            }

            Vec_t tmpacc =
                ComputeAccelerationSoftened(currentPos[kDims], s0, s1);

            // Write to buffer
            if (j != kN - 1) {
              for (int s = 0; s < kDims; s++) {
                acc[k][l][s] = acc[k][l][s] + tmpacc[s];
              }
            } else {
              for (int s = 0; s < kDims; s++) {
                // Writeout
                Data_t v = (acc[k][l][s] + tmpacc[s]) * kTimestep;
                float vel = velocity[i * (kDepthProcessingElement * kTileSize) +
                                     kDepthProcessingElement * k + l][s] +
                            v;
                velocityNew[i * (kDepthProcessingElement * kTileSize) +
                            kDepthProcessingElement * k + l][s] = vel;
                positionMassNew[i * (kDepthProcessingElement * kTileSize) +
                                kDepthProcessingElement * k + l][s] =
                    positionMass[i * (kDepthProcessingElement * kTileSize) +
                                 kDepthProcessingElement * k + l][s] +
                    vel * kTimestep;

                // reset acc
                acc[k][l][s] = 0.0;
              }
            }
          }
        }
      }
    }
    // use swap buffers?
    for (int i = 0; i < kN; i++) {
      velocity[i] = velocityNew[i];
      positionMassNew[i] = positionMass[i];
    }
  }
}
