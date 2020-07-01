/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <iostream>
#include "NBody.h"

void PrintUsage() {
  std::cerr << "Usage: ./Stats <timesteps> [<routed frequency>]\n" << std::flush;
}

// Prints the expected performance of the current configuration in hardware.
// If a different frequency is achieved, it can be passed as the first argument
// to the executable.
int main(int argc, char **argv) {
  if (argc > 3 || argc < 2) {
    PrintUsage();
    return 1;
  }
  const auto timesteps = std::stoul(argv[1]);
  const float frequency = (argc < 3) ? kFrequency : std::stof(argv[2]);
  const double nOps = NumberOfOps(timesteps);
  std::cout << "Frequency:            " << frequency << " MHz\n";
  std::cout << "Number of operations: " << nOps << "\n";
  const double expected_runtime =
      timesteps *
      (kUnrollDepth * kPipelineFactor +
       kNTiles * (static_cast<double>(kNBodies) * kPipelineFactor +
                  kUnrollDepth * kPipelineFactor)) /
      (1e6 * frequency);
  const auto expected_perf = 1e-9 * nOps / expected_runtime;
  const auto peak_perf = 1e-9 * (kUnrollDepth * 22 * (1e6 * frequency));
  std::cout << "Expected runtime:     " << expected_runtime << " seconds\n";
  std::cout << "Peak runtime:         " << nOps / (1e9 * peak_perf)
            << " seconds\n";
  std::cout << "Expected performance: " << expected_perf << " GOp/s\n";
  std::cout << "Peak performance:     " << peak_perf << " GOp/s\n";
  return 0;
}
