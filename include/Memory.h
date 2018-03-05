/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "NBody.h"
#include "hlslib/Stream.h"


void ReadMemory_PositionMass(MemoryPack_t const memory[],
                             hlslib::Stream<MemoryPack_t> &pipe);

void ContractWidth_PositionMass(hlslib::Stream<MemoryPack_t> &wide,
                                hlslib::Stream<PosMass_t> &narrow);

void ExpandWidth_PositionMass(hlslib::Stream<PosMass_t> &narrow,
                              hlslib::Stream<MemoryPack_t> &wide);

void WriteMemory_PositionMass(hlslib::Stream<MemoryPack_t> &pipe,
                              MemoryPack_t memory[]);

void RepeatFirstTile(hlslib::Stream<PosMass_t> &streamIn,
                     hlslib::Stream<PosMass_t> &streamOut);

void ReadMemory_Velocity(MemoryPack_t const memory[],
                         hlslib::Stream<MemoryPack_t> &pipe);

// Used for testing software. Does not work with AXI if kDims is 3 
void ReadSingle_Velocity(Vec_t const memory[],
                         hlslib::Stream<Vec_t> &pipe);

void WriteMemory_Velocity(hlslib::Stream<MemoryPack_t> &pipe,
                          MemoryPack_t memory[]);

// Used for testing software. Does not work with AXI if kDims is 3 
void WriteSingle_Velocity(hlslib::Stream<Vec_t> &pipe,
                          Vec_t memory[]);
