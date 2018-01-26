/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "NBody.h"
#include "hlslib/Stream.h"

void ReadVector(Vec_t const memory[], hlslib::Stream<Vec_t> &stream);

void WriteVector(hlslib::Stream<Vec_t> &stream, Vec_t memory[]);

void ReadMemory(Data_t const memory[], hlslib::Stream<Data_t> &stream);

void PackData(hlslib::Stream<Vec_t> &posIn, hlslib::Stream<Data_t> &massIn,
              hlslib::Stream<Packed> &packedOut);
