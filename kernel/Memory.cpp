/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 
//
// #include "Memory.h"
//
// void ReadVector(Vec_t const memory[], hlslib::Stream<Vec_t> &stream) {
// Time:
//   for (int t = 0; t < kSteps; ++t) {
//   Outer:
//     for (int n = 0; n < kN / kTileSize; ++n) {
//     Inner:
//       for (int m = 0; m < kN; ++m) {
//         #pragma HLS LOOP_FLATTEN
//         #pragma HLS PIPELINE II=1
//         hlslib::WriteBlocking(stream, memory[m]);
//       }
//     }
//   }
// }
//
// void WriteVector(hlslib::Stream<Vec_t> &stream, Vec_t memory[]) {
// Time:
//   for (int t = 0; t < kSteps; ++t) {
//   Outer:
//     for (int n = 0; n < kN / kTileSize; ++n) {
//     Inner:
//       for (int m = 0; m < kN; ++m) {
//         #pragma HLS LOOP_FLATTEN
//         #pragma HLS PIPELINE II=1
//         memory[m] = hlslib::ReadBlocking(stream);
//       }
//     }
//   }
// }
//
// void ReadMemory(Data_t const memory[],
//                 hlslib::Stream<Data_t> &stream) {
//   for (int t = 0; t < kSteps; ++t) {
//   Outer:
//     for (int n = 0; n < kN / kTileSize; ++n) {
//     Inner:
//       for (int m = 0; m < kN; ++m) {
//         #pragma HLS LOOP_FLATTEN
//         #pragma HLS PIPELINE II=1
//         hlslib::WriteBlocking(stream, memory[m]);
//       }
//     }
//   }
// }
//
// void PackData(hlslib::Stream<Vec_t> &posIn,
//               hlslib::Stream<Data_t> &massIn,
//               hlslib::Stream<Packed> &packedOut) {
//   for (int t = 0; t < kSteps; ++t) {
//   Outer:
//     for (int n = 0; n < kN / kTileSize; ++n) {
//     Inner:
//       for (int m = 0; m < kN; ++m) {
//         #pragma HLS LOOP_FLATTEN
//         #pragma HLS PIPELINE II=1
//         const Packed packed(Vec_t{static_cast<Data_t>(0)},
//                             hlslib::ReadBlocking(posIn),
//                             hlslib::ReadBlocking(massIn));
//         hlslib::WriteBlocking(packedOut, packed); 
//       }
//     }
//   }
// }
