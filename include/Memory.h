/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "NBody.h"
#include "hlslib/Stream.h"


void ReadMemory_PositionMass(MemoryPack_t const memory[],
                             hlslib::Stream<MemoryPack_t> &pipe);

void WriteMemory_PositionMass(hlslib::Stream<MemoryPack_t> &pipe,
                              MemoryPack_t memory[]);

void RepeatFirstTile(hlslib::Stream<PosMass_t> &streamIn,
                     hlslib::Stream<PosMass_t> &streamOut);

void ReadMemory_Velocity(MemoryPack_t const memory[],
                         hlslib::Stream<MemoryPack_t> &pipe);

void WriteMemory_Velocity(hlslib::Stream<MemoryPack_t> &pipe,
                          MemoryPack_t memory[]);

#ifndef HLSLIB_SYNTHESIS

void ContractWidth_PositionMass(hlslib::Stream<MemoryPack_t> &wide,
                                hlslib::Stream<PosMass_t> &narrow);

void ExpandWidth_PositionMass(hlslib::Stream<PosMass_t> &narrow,
                              hlslib::Stream<MemoryPack_t> &wide);

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
template <typename T, unsigned width>
void ConvertMemoryToNonDivisible(
    hlslib::Stream<MemoryPack_t> &streamIn,
    hlslib::Stream<hlslib::DataPack<T, width>> &streamOut, int iterations) {
  int dataRemaining = 0;
  MemoryPack_t curr;
  MemoryPack_t next;
  for (unsigned i = 0; i < iterations; ++i) {
    hlslib::DataPack<T, width> out;
    if (dataRemaining == 0) {
      curr = streamIn.Pop();
      void * pnt = (void*) &curr;
      for(int j = 0; j < width; j++){
        out[j] = curr[j];
      }
      dataRemaining = sizeof(MemoryPack_t)/sizeof(Data_t) - width;
    } else if (dataRemaining < width) {
      next = streamIn.Pop();
      for(int j = 0; j < width; j++){
        if(j < dataRemaining){
          out[j] = curr[j +  sizeof(MemoryPack_t)/sizeof(Data_t) - dataRemaining];
        }else{
          out[j] = next[j - dataRemaining];
        }
      }
      curr = next;
      dataRemaining = sizeof(MemoryPack_t)/sizeof(Data_t) - width + dataRemaining;
    } else { // bytesRemaining >= kOutBytes
      for(int j = 0; j < width; j++){
        out[j] = curr[j +  sizeof(MemoryPack_t)/sizeof(Data_t) - dataRemaining];
      }
      dataRemaining = dataRemaining - width;
    }
    streamOut.Push(out);
  }
}

/// Takes a stream of wide memory accesses and converts it into elements of a
/// size that do not divide into the memory width. When crossing a memory
/// boundary, use bytes from both the previous and next memory access.
template <typename T, unsigned width>
void ConvertNonDivisibleToMemory(
    hlslib::Stream<hlslib::DataPack<T, width>> &streamIn,
    hlslib::Stream<MemoryPack_t> &streamOut, unsigned iterations) {
    int currentPos = 0;
    const int size = sizeof(MemoryPack_t)/sizeof(Data_t);
    MemoryPack_t out;
    MemoryPack_t out2;
    hlslib::DataPack<T, width> last;
  for (unsigned i = 0; i < iterations; ++i) {
    hlslib::DataPack<T, width> readT = streamIn.Pop();
    if(size - currentPos > width){
      for(int j = 0; j < width; j++){
        out[currentPos++] = readT[j];
      }
    }else{
      int offset = size - currentPos;
      for(int j = 0; j < width; j++){
        if(j < offset){
          out[currentPos++] = readT[j];
        }else{
          out2[currentPos - size] = readT[j];
          currentPos++;
        }
      }
      streamOut.Push(out);
      out = out2;
      currentPos = currentPos - size;
    }
  }
}

#endif
