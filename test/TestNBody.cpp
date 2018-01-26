/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "NBody.h"

int main() {
  std::vector<Data_t> mass(kN);
  std::vector<Vec_t> velocity(kN);
  std::vector<Vec_t> position(kN);
  NBody(mass.data(), position.data(), position.data(), velocity.data(),
        velocity.data());
  return 0;
}
