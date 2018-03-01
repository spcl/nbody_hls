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
    Vec_t v0new[kN];
    Vec_t s0new[kN];
      for (int n = 0; n < kN; ++n) {
        const auto v0 = velocity[n];
        const auto s0 = position[n];
        Vec_t a;
        a[0] = 0.0;
        a[1] = 0.0;
        a[2] = 0.0;
        for (int m = 0; m < kN; ++m) {
          // if (n == m) continue;
          const auto v1 = velocity[m];
          const auto s1 = position[m];
          const auto m1 = mass[m];
          a += ComputeAccelerationSoftened(m1, s0, s1);
        }
        for(int d = 0; d < kDims; d++){
        v0new[n][d] = velocity[n][d] + a[d] * kTimestep;
        s0new[n][d] = position[n][d] + v0new[n][d] * kTimestep;
        }
      }
      for(int i = 0; i < kN; i++){
      velocity[i] = v0new[i];
      position[i] = s0new[i];
    }
  }
}

void NewAlgorithm(hlslib::DataPack<Data_t, kDims + 1> positionMass[], Vec_t velocity[]){
  for(int t = 0; t < kSteps; t++){
    hlslib::DataPack<Data_t, kDims + 1> positionMassNew[kN];
    Vec_t velocityNew[kN];
    //is a datapack initialised to zero by default?
    hlslib::DataPack<Data_t, kDims + 1> posWeight[2][kTileSize][kDepthProcessingElement];
    Vec_t acc[kTileSize][kDepthProcessingElement];
    int next = 0;

    for(int i = 0; i < kTileSize; i++){
      for(int j = 0; j < kDepthProcessingElement; j++){
        for(int k = 0; k < kDims + 1; k++){
          posWeight[0][i][j][k] = positionMass[i*kDepthProcessingElement+j][k];
          if(k != kDims) acc[i][j][k] = 0.0;
        }
      }
    }
    for(int i = 0; i < kN/(kDepthProcessingElement*kTileSize); i++){
      next = 1 - next;
      for(int j = 0; j < kN; j++){
        hlslib::DataPack<Data_t, kDims + 1> currentPos;
        for(int k = 0; k < kDims + 1; k++){
          currentPos[k] = positionMass[j][k];

          //Here I populate the next buffer with appropriate elments.
          if(j >= (i+1)*kDepthProcessingElement*kTileSize && j < (i + 2)*kDepthProcessingElement*kTileSize && i != kN/(kDepthProcessingElement*kTileSize) - 1){
            int a = j - (i + 1)*kDepthProcessingElement*kTileSize;
            posWeight[next][a / kDepthProcessingElement][a % kDepthProcessingElement][k] = currentPos[k];
          }
        }
          //Now comes the second unroll, this time per processing element
          for(int l = 0; l < kDepthProcessingElement; l++){

            //The loop that is replicated in Hardware
            for(int k = 0; k < kTileSize; k++){
              Vec_t s0;
              Vec_t s1;

              //Accounts for the fact that ComputeAccelerationSoftened does not take posWeight args yet
              for(int s = 0; s < kDims; s++){
                s0[s] = posWeight[1-next][k][l][s];
                s1[s] = currentPos[s];
              }

              Vec_t tmpacc = ComputeAccelerationSoftened(currentPos[kDims], s0, s1);

              //Write to buffer
              if(j != kN - 1){
                for(int s = 0; s < kDims; s++){
                  acc[k][l][s] = acc[k][l][s] + tmpacc[s];
                }
              }else{
                for(int s = 0; s < kDims; s++){

                  //Writeout
                  Data_t v = (acc[k][l][s] + tmpacc[s]) * kTimestep;
                  float vel = velocity[i*(kDepthProcessingElement*kTileSize) + kDepthProcessingElement*k + l][s] + v;
                  velocityNew[i*(kDepthProcessingElement*kTileSize) + kDepthProcessingElement*k + l][s] = vel;
                  positionMassNew[i*(kDepthProcessingElement*kTileSize) + kDepthProcessingElement*k + l][s] = positionMass[i*(kDepthProcessingElement*kTileSize) + kDepthProcessingElement*k + l][s] + vel*kTimestep;

                  //reset acc
                  acc[k][l][s] = 0.0;
                }
              }
            }
          }
        }
      }
      //use swap buffers?
      for(int i = 0; i < kN; i++){
        velocity[i] = velocityNew[i];
        positionMassNew[i] = positionMass[i];
      }
  }
}

void interaction(Vec_t p1, Vec_t p2, Data_t m2, Vec_t *acc){
  Vec_t r;
  r[0] = p2[0] - p1[0];
  r[1] = p2[1] - p1[1];
  r[2] = p2[2] - p1[2];

  Data_t d2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + kEps2;

  Data_t d6 = d2 * d2 * d2;

  Data_t fac = 1.0f/sqrt(d6); //they cast to double in one file here

  Data_t s = m2 * fac;

  (*acc)[0] = (*acc)[0] + r[0] * s;
  (*acc)[1] = (*acc)[1] + r[1] * s;
  (*acc)[2] = (*acc)[2] + r[2] * s;
 }

void computeGraviation(Data_t const mass[], Vec_t position[], Vec_t force[]){
  for(int i = 0; i < kN; i++){
    Vec_t acc;
    acc[0] = 0.0;
    acc[1] = 0.0;
    acc[2] = 0.0;
    for(int j = 0; j < kN; j++){
      interaction(position[i], position[j], mass[j], &acc);
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

      // Data_t invm = mass[i]; //invmass??, they do not invert??
      Data_t invm = 1.0; //I really don't get this, I have to look into their impl. again.
      //They always call this with invm = 1.0, but I don't get why you would do this.

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
