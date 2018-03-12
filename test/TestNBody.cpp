/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <cmath>
#include "NBody.h"
#include "Utility.h"

void NewAlgorithmReference(PosMass_t positionMass[],
                           Vec_t velocity[]);

int main() {
  std::vector<Vec_t> velocity(kNBodies);
  std::vector<PosMass_t> position(2 * kNBodies); // Double buffering

  // maybe make those configurable in the future :), if we use them
  float clusterScale = 1.54f;
  float velocityScale = 8.0f;

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> distMass(1e10, 1e12);
  std::uniform_real_distribution<double> distVelocity(-1e3, 1e3);
  std::uniform_real_distribution<double> distPosition(-1e6, 1e6);

  float scale = clusterScale * std::max<Data_t>(1.0, kNBodies / (1024.0));
  float vscale = velocityScale * scale;

  int i = 0;

  while (i < kNBodies) {
    Vec_t point;
    float lenSqr = 0.0;
    for (int j = 0; j < kDims; j++) {
      point[j] = rand() / (float)RAND_MAX * 2 - 1;
      lenSqr += point[j] * point[j];
    }
    if (lenSqr > 1) continue;

    Vec_t vel;
    lenSqr = 0.0;
    for (int j = 0; j < kDims; j++) {
      vel[j] = rand() / (float)RAND_MAX * 2 - 1;
      lenSqr += vel[j] * vel[j];
    }
    if (lenSqr > 1) continue;

    for (int j = 0; j < kDims; j++) {
      position[i][j] = point[j] * scale;
      velocity[i][j] = vel[j] * vscale;
    }

    position[i][kDims] = 1.0f;  // mass

    i++;
  }

  // NBody(reinterpret_cast<MemoryPack_t const *>(&mass[0]),
  //       reinterpret_cast<MemoryPack_t const *>(&position[0]),
  //       reinterpret_cast<MemoryPack_t *>(&position[0]),
  //       reinterpret_cast<MemoryPack_t const *>(&velocity[0]),
  //       reinterpret_cast<MemoryPack_t *>(&velocity[0]));

  std::vector<PosMass_t> positionRef(position);
  std::vector<Vec_t> velocityRef(velocity);
  std::vector<PosMass_t> positionHardware(position);
  std::vector<Vec_t> velocityHardware(velocity);

  hlslib::Stream<MemoryPack_t> velocityReadMemory("velocityReadMemory");
  hlslib::Stream<MemoryPack_t> velocityWriteMemory("velocityWriteMemory");
  hlslib::Stream<Vec_t> velocityReadKernel("velocityReadKernel");
  hlslib::Stream<Vec_t> velocityWriteKernel("velocityWriteKernel");

  std::cout << "Running reference implementation of new algorithm..."
            << std::flush;
  NewAlgorithmReference(position.data(), velocity.data());
  std::cout << " Done.\n";

  std::cout << "Running CUDA reference implementation..." << std::flush;
  ReferenceLikeCUDA(positionRef.data(), velocityRef.data());
  std::cout << " Done.\n";

  std::cout << "Running emulation of hardware implementation..." << std::flush;
  NBody(reinterpret_cast<MemoryPack_t const *>(&positionHardware[0]),
        reinterpret_cast<MemoryPack_t *>(&positionHardware[0]),
        reinterpret_cast<MemoryPack_t const *>(&velocityHardware[0]),
        reinterpret_cast<MemoryPack_t *>(&velocityHardware[0]),
        velocityReadMemory, velocityReadKernel, velocityWriteKernel,
        velocityWriteMemory);
  std::cout << " Done.\n";

  std::cout << "Verifying results..." << std::flush;
  constexpr int kPrintBodies = 20;
  for (int i = 0; i < kNBodies; ++i) {
    if (i < kPrintBodies) {
      std::cout << position[i] << " / " << positionRef[i] << ", "
                << velocity[i] << " / " << velocityRef[i] << "\n";
    }
    for (int d = 0; d < kDims; ++d) {
      {
        const auto diff = std::abs(position[i][d] - positionRef[i][d]);
        if (diff >= 1e-4) {
          std::cerr << "Mismatch in reference implementation at index " << i
                    << ": " << position[i] << " (should be " << positionRef[i]
                    << ")." << std::endl;
          return 1;
        }
      }
      {
        const auto diff = std::abs(position[i][d] - positionHardware[i][d]);
        if (diff >= 1e-4) {
          std::cerr << "Mismatch in hardware implementation at index " << i
                    << ": " << positionHardware[i] << " (should be "
                    << position[i] << ")." << std::endl;
          return 1;
        }
      }
    }
  }
  std::cout << " Done." << std::endl;

  return 0;
}

// Reference implementation for new hardware design
void NewAlgorithmReference(PosMass_t positionMass[], Vec_t velocity[]) {
  for (int t = 0; t < kSteps; t++) {
    PosMass_t positionMassNew[kNBodies];
    Vec_t velocityNew[kNBodies];
    // is a datapack initialised to zero by default?
    PosMass_t posWeight[2][kUnrollDepth][kPipelineFactor];
    Vec_t acc[kUnrollDepth][kPipelineFactor];
    int next = 0;

    for (int i = 0; i < kUnrollDepth; i++) {
      for (int j = 0; j < kPipelineFactor; j++) {
        for (int k = 0; k < kDims + 1; k++) {
          posWeight[0][i][j][k] = positionMass[i * kPipelineFactor + j][k];
          if (k != kDims) acc[i][j][k] = 0.0;
        }
      }
    }
    for (int i = 0; i < kNBodies / (kPipelineFactor * kUnrollDepth); i++) {
      next = 1 - next;
      for (int j = 0; j < kNBodies; j++) {
        PosMass_t currentPos;
        for (int k = 0; k < kDims + 1; k++) {
          currentPos[k] = positionMass[j][k];

          // Here I populate the next buffer with appropriate elments.
          if (j >= (i + 1) * kPipelineFactor * kUnrollDepth &&
              j < (i + 2) * kPipelineFactor * kUnrollDepth &&
              i != kNBodies / (kPipelineFactor * kUnrollDepth) - 1) {
            int a = j - (i + 1) * kPipelineFactor * kUnrollDepth;
            posWeight[next][a / kPipelineFactor][a % kPipelineFactor][k] =
                currentPos[k];
          }
        }

        // Now comes the second unroll, this time per processing element
        for (int l = 0; l < kPipelineFactor; l++) {
          // The loop that is replicated in Hardware
          for (int k = 0; k < kUnrollDepth; k++) {
            PosMass_t s0 = posWeight[1 - next][k][l];
            Vec_t tmpacc = ComputeAcceleration<true>(s0, currentPos);
            // Write to buffer
            if (j != kNBodies - 1) {
              acc[k][l] = acc[k][l] + tmpacc;
            } else {
              for (int s = 0; s < kDims; s++) {
                // Writeout
                Data_t v = (acc[k][l][s] + tmpacc[s]) * kTimestep;
                float vel = velocity[i * (kPipelineFactor * kUnrollDepth) +
                                     kPipelineFactor * k + l][s] +
                            v;
                velocityNew[i * (kPipelineFactor * kUnrollDepth) +
                            kPipelineFactor * k + l][s] = vel;
                positionMassNew[i * (kPipelineFactor * kUnrollDepth) +
                                kPipelineFactor * k + l][s] =
                    positionMass[i * (kPipelineFactor * kUnrollDepth) +
                                 kPipelineFactor * k + l][s] +
                    vel * kTimestep;
                // reset acc
                acc[k][l][s] = 0.0;
              }
              positionMassNew[i * (kPipelineFactor * kUnrollDepth) +
                              kPipelineFactor * k + l][kDims] = positionMass[i * (kPipelineFactor * kUnrollDepth) +
                                              kPipelineFactor * k + l][kDims];
            }
          }
        }
      }
    }
    // use swap buffers?
    for (int i = 0; i < kNBodies; i++) {
      velocity[i] = velocityNew[i];
      positionMass[i] = positionMassNew[i];
    }
  }
}
