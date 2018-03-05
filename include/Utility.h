/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      June 2017
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include <cstddef>
#include <iterator>
#include <vector>
#include "NBody.h"

void Reference(PosMass_t position[], Vec_t velocity[]) {
  for (int t = 0; t < kSteps; ++t) {
    Vec_t v0new[kNBodies];
    PosMass_t s0new[kNBodies];
    for (int n = 0; n < kNBodies; ++n) {
      const auto v0 = velocity[n];
      const auto s0 = position[n];
      Vec_t a;
      a[0] = 0.0;
      a[1] = 0.0;
      a[2] = 0.0;
      for (int m = 0; m < kNBodies; ++m) {
        // if (n == m) continue;
        const auto v1 = velocity[m];
        const auto s1 = position[m];
        a += ComputeAcceleration<true>(s0, s1);
      }
      for (int d = 0; d < kDims; d++) {
        v0new[n][d] = velocity[n][d] + a[d] * kTimestep;
        s0new[n][d] = position[n][d] + v0new[n][d] * kTimestep;
      }
      s0new[n][kDims] = s0[kDims];
    }
    for (int i = 0; i < kNBodies; i++) {
      velocity[i] = v0new[i];
      position[i] = s0new[i];
    }
  }
}

inline void Interaction(PosMass_t const &p1, PosMass_t const &p2, Vec_t *acc) {
  Vec_t r;
  r[0] = p2[0] - p1[0];
  r[1] = p2[1] - p1[1];
  r[2] = p2[2] - p1[2];

  Data_t d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + kSoftening;

  Data_t d6 = d2 * d2 * d2;

  Data_t fac = 1.0f / sqrt(d6);  // they cast to double in one file here

  Data_t s = p2[3] * fac;

  (*acc)[0] = (*acc)[0] + r[0] * s;
  (*acc)[1] = (*acc)[1] + r[1] * s;
  (*acc)[2] = (*acc)[2] + r[2] * s;
}

inline void ComputeGravitation(PosMass_t position[], Vec_t force[]) {
  for (int i = 0; i < kNBodies; i++) {
    Vec_t acc;
    acc[0] = 0.0;
    acc[1] = 0.0;
    acc[2] = 0.0;
    for (int j = 0; j < kNBodies; j++) {
      Interaction(position[i], position[j], &acc);
    }
    force[i][0] = acc[0];
    force[i][1] = acc[1];
    force[i][2] = acc[2];
  }
}

inline void ReferenceLikeCUDA(PosMass_t position[], Vec_t velocity[]) {
  for (int j = 0; j < kSteps; j++) {
    Vec_t force[kNBodies];
    ComputeGravitation(position, force);

    for (int i = 0; i < kNBodies; i++) {
      Vec_t pos;
      Vec_t vel;
      Vec_t f;

      pos[0] = position[i][0];
      pos[1] = position[i][1];
      pos[2] = position[i][2];

      // Data_t invm = mass[i]; //invmass??, they do not invert??
      Data_t invm = 1.0;  // I really don't get this, I have to look into their
                          // impl. again.
      // They always call this with invm = 1.0, but I don't get why you would do
      // this.

      vel[0] = velocity[i][0];
      vel[1] = velocity[i][1];
      vel[2] = velocity[i][2];

      f[0] = force[i][0];
      f[1] = force[i][1];
      f[2] = force[i][2];

      vel[0] = vel[0] + (f[0] * invm) * kTimestep;
      vel[1] = vel[1] + (f[1] * invm) * kTimestep;
      vel[2] = vel[2] + (f[2] * invm) * kTimestep;

      // vel[0] = vel[0] * kDamping;
      // vel[1] = vel[1] * kDamping;
      // vel[2] = vel[2] * kDamping;

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
