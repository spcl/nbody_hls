/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "NBody.h"
#include "Utility.h"
#include <cmath>

int main() {
  std::vector<Data_t> mass(kN);
  std::vector<Vec_t> velocity(kN);
  std::vector<Vec_t> position(kN);
  std::vector<Vec_t> velocityRef(kN);
  std::vector<Vec_t> positionRef(kN);

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> distMass(1e-12, 1e-10);
  std::uniform_real_distribution<double> distVelocity(-1e3, 1e3);
  std::uniform_real_distribution<double> distPosition(-1e7, 1e7);

  std::for_each(mass.begin(), mass.end(),
                [&distMass, &rng](Data_t &in) { in = distMass(rng); });
  std::for_each(velocity.begin(), velocity.end(),
                [&distVelocity, &rng](Vec_t &in) {
                  for (int d = 0; d < kDims; ++d) {
                    in[d] = distVelocity(rng);
                  }
                });
  std::for_each(position.begin(), position.end(),
                [&distPosition, &rng](Vec_t &in) {
                  for (int d = 0; d < kDims; ++d) {
                    in[d] = distPosition(rng);
                  }
                });

  NBody(reinterpret_cast<MemoryPack_t const *>(&mass[0]),
        reinterpret_cast<MemoryPack_t const *>(&position[0]),
        reinterpret_cast<MemoryPack_t *>(&position[0]),
        reinterpret_cast<MemoryPack_t const *>(&velocity[0]),
        reinterpret_cast<MemoryPack_t *>(&velocity[0]));

  Reference(mass.data(), positionRef.data(), velocityRef.data());

  for (unsigned i = 0; i < std::min<unsigned>(kN, 10); ++i) {
    std::cout << position[i] << " / " << positionRef[i] << ", " << velocity[i]
              << ", " << velocityRef[i] << "\n";
  }

  return 0;
}
