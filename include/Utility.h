/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <cstddef>
#include <iterator>
#include <vector>
#include "NBody.h"

void Reference(Data_t const mass[], Vec_t position[], Vec_t velocity[]) {
  for (int t = 0; t < kSteps; ++t) {
    for (int d = 0; d < kDims; ++d) {
      for (int n = 0; n < kN; ++n) {
        const auto v0 = velocity[n];
        const auto s0 = position[n];
        Vec_t a(static_cast<Data_t>(0));
        for (int m = 0; m < kN; ++m) {
          if (n == m) continue;
          const auto v1 = velocity[m];
          const auto s1 = position[m];
          const auto m1 = mass[m];
          a += ComputeAcceleration(m1, s0, s1); 
        }
        const auto v0new = v0 + a * kTimestep;
        const auto s0new = s0 + v0 * kTimestep;
        velocity[n] = v0new;
        position[n] = s0new;
      }
    }
  }
}
