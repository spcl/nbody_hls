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
          a += ComputeAccelerationSoftened(m1, s0, s1);
        }
        const auto v0new = v0 + a * kTimestep;
        const auto s0new = s0 + v0 * kTimestep;
        velocity[n] = v0new;
        position[n] = s0new;
      }
    }
  }
}

void interaction(Vec_t p1, Vec_t p2, Data_t m2, Vec_t acc){
  Vec_t r;
  r[0] = p2[0] - p1[0];
  r[1] = p2[1] - p1[1];
  r[2] = p2[2] - p1[2];

  float d2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + kEps2;

  float d6 = d2 * d2 * d2;

  float fac = 1.0f/sqrtf(d6);

  float s = m2 * fac;

  acc[0] = acc[0] + r[0] * s;
  acc[1] = acc[1] + r[1] * s;
  acc[2] = acc[2] + r[2] * s;
 }

void computeGraviation(Data_t const mass[], Vec_t position[], Vec_t force[]){
  for(int i = 0; i < kN; i++){
    Vec_t acc;
    for(int j = 0; j < kN; j++){
      interaction(position[i], position[j], mass[j], acc);
    }
    force[i][0] = acc[0];
    force[i][1] = acc[1];
    force[i][2] = acc[2];
  }
}

void ReferenceLikeCUDA(Data_t const mass[], Vec_t position[], Vec_t velocity[]){
  for(int j = 0; j < kSteps; j++){
    Vec_t force[kN];
    computeGraviation(mass, position, force);

    for(int i = 0; i < kN; i++){
      Vec_t pos;
      Vec_t vel;
      Vec_t f;

      pos[0] = position[i][0];
      pos[1] = position[i][1];
      pos[2] = position[i][2];

      Data_t invm = 1.0/mass[i]; //invmass??

      vel[0] = velocity[i][0];
      vel[1] = velocity[i][1];
      vel[2] = velocity[i][2];

      f[0] = force[i][0];
      f[1] = force[i][1];
      f[2] = force[i][2];

      vel[0] = vel[0] + (f[0] * invm) * kTimestep;
      vel[1] = vel[1] + (f[1] * invm) * kTimestep;
      vel[2] = vel[2] + (f[2] * invm) * kTimestep;

      vel[0] = vel[0] * kDamping;
      vel[1] = vel[1] * kDamping;
      vel[2] = vel[2] * kDamping;

      pos[0] = pos[0] + vel[0] * kTimestep;
      pos[1] = pos[1] + vel[1] * kTimestep;
      pos[2] = pos[2] + vel[2] * kTimestep;

      position[i][0] = pos[0];
      position[i][1] = pos[1];
      position[i][2] = pos[2];

      velocity[i][0] = vel[0];
      velocity[i][1] = vel[1];
      velocity[i][2] = vel[2];
    }
  }
}
