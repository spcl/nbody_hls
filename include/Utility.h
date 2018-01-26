/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <cstddef>
#include <iterator>
#include <vector>
#include "NBody.h"

void Reference(Data_t const mass[], Data_t position[], Data_t velocity[]) {
  for (int t = 0; t < kNTimesteps; ++t) {
    for (int d = 0; d < kDims; ++d) {
      for (int n = 0; n < kN; ++n) {
        const auto v0 = velocity[d*kN + n];
        const auto s0 = position[d*kN + n];
        Data_t a = 0;
        for (int m = 0; m < kN; ++m) {
          const auto v1 = velocity[d*kN + m];
          const auto s1 = position[d*kN + m];
          const auto m1 = mass[d*kN + m];
          a += ComputeAcceleration(m1, s0, s1); 
        }
        const auto v0new = v0 + a * kTimestep;
        const auto s0new = s0 + v0 * kTimestep;
        velocity[d*kN + n] = v0new;
        position[d*kN + n] = s0new;
      }
    }
  }
}
