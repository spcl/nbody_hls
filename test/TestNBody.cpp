/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "NBody.h"
#include "Utility.h"
#include <cmath>

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

  NewAlgorithm(posWeight.data(), velocity.data());
  // Reference(mass.data(), position.data(), velocity.data());
  ReferenceLikeCUDA(mass.data(), positionRef.data(), velocityRef.data());

  for (unsigned i = 0; i < std::min<unsigned>(kN, 100); ++i) {
    std::cout << posWeight[i] << " / " << positionRef[i] << ", " << velocity[i]
              << "/ " << velocityRef[i] << "\n";
    if(i == 4*16 -1){
      std::cout << "\n\n";
    }
  }

  return 0;
}
