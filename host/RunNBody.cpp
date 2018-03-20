/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "NBody.h"
#include "hlslib/SDAccel.h"

int main(int argc, char **argv) {

  std::vector<Vec_t> velocity(kNBodies);
  std::vector<PosMass_t> position(2 * kNBodies); // Double buffering

  // maybe make those configurable in the future :), if we use them
  float clusterScale = 1.54f;
  float velocityScale = 8.0f;

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> distMass(1e10, 1e12);
  std::uniform_real_distribution<double> distVelocity(-1e3, 1e3);
  std::uniform_real_distribution<double> distPosition(-1e6, 1e6);

  float scale = clusterScale * std::max<Data_t>(1.0, kNBodies / (1024.0));
  float vscale = velocityScale * scale;

  int i = 0;

  while (i < kNBodies) {
    Vec_t point;
    float lenSqr = 0.0;
    for (int j = 0; j < kDims; j++) {
      point[j] = rand() / (float)RAND_MAX * 2 - 1;
      lenSqr += point[j] * point[j];
    }
    if (lenSqr > 1) continue;

    Vec_t vel;
    lenSqr = 0.0;
    for (int j = 0; j < kDims; j++) {
      vel[j] = rand() / (float)RAND_MAX * 2 - 1;
      lenSqr += vel[j] * vel[j];
    }
    if (lenSqr > 1) continue;

    for (int j = 0; j < kDims; j++) {
      position[i][j] = point[j] * scale;
      velocity[i][j] = vel[j] * vscale;
    }

    position[i][kDims] = 1.0f;  // mass

    i++;
  }

  // NBody(reinterpret_cast<MemoryPack_t const *>(&mass[0]),
  //       reinterpret_cast<MemoryPack_t const *>(&position[0]),
  //       reinterpret_cast<MemoryPack_t *>(&position[0]),
  //       reinterpret_cast<MemoryPack_t const *>(&velocity[0]),
  //       reinterpret_cast<MemoryPack_t *>(&velocity[0]));

  std::vector<PosMass_t> positionRef(position);
  std::vector<Vec_t> velocityRef(velocity);
  std::vector<PosMass_t> positionHardware(position);
  std::vector<Vec_t> velocityHardware(velocity);

  hlslib::Stream<MemoryPack_t> velocityReadMemory("velocityReadMemory");
  hlslib::Stream<MemoryPack_t> velocityWriteMemory("velocityWriteMemory");
  hlslib::Stream<Vec_t> velocityReadKernel("velocityReadKernel");
  hlslib::Stream<Vec_t> velocityWriteKernel("velocityWriteKernel");

  // std::cout << "Running reference implementation of new algorithm..."
  //           << std::flush;
  // NewAlgorithmReference(position.data(), velocity.data());
  // std::cout << " Done.\n";
  //
  // std::cout << "Running CUDA reference implementation..." << std::flush;
  // ReferenceLikeCUDA(positionRef.data(), velocityRef.data());
  // std::cout << " Done.\n";

  try {

    std::cout << "Initializing OpenCL context..." << std::flush;
    hlslib::ocl::Context context;
    std::cout << " Done.\n";

    std::cout << "Initializing device memory..." << std::flush;
    auto positionDevice =
        context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::readWrite>(
            hlslib::ocl::MemoryBank::bank0, kMemorySizePosition);
    auto velocityDevice =
        context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::readWrite>(
            hlslib::ocl::MemoryBank::bank1, kMemorySizeVelocity);
    std::cout << " Done.\n";

    auto startWithMemory = std::chrono::high_resolution_clock::now();
    std::cout << "Copying data to device..." << std::flush;
    positionDevice.CopyFromHost(
        reinterpret_cast<MemoryPack_t const *>(&positionHardware[0]));
    velocityDevice.CopyFromHost(
        reinterpret_cast<MemoryPack_t const *>(&velocityHardware[0]));
    std::cout << " Done.\n";

    std::cout << "Programming device..." << std::flush;
    auto kernel =
        context.MakeKernel("NBody.xclbin", "nbody_kernel", positionDevice,
                           positionDevice, velocityDevice, velocityDevice);
    std::cout << " Done.\n";

    std::cout << "Executing kernel..." << std::flush;
    const auto elapsed = kernel.ExecuteTask();
    std::cout << " Done.\n";

    // const auto perf = 1e-9 *
    //                   (2 * static_cast<float>(kSizeN) * kSizeM * kSizeP) /
    //                   elapsed.first;

    std::cout << "Copying back result..." << std::flush;
    positionDevice.CopyToHost(
        reinterpret_cast<MemoryPack_t *>(&positionHardware[0]));
    std::cout << " Done.\n";

    auto endWithMemory = std::chrono::high_resolution_clock::now();
    auto elapsedWithMemory = (endWithMemory - startWithMemory).count();

    std::cout << "Kernel executed in " << elapsed.first << " seconds.\n";
    // corresponding to a performance of " << perf
    // << " GOp/s (" << perfWithMemory
    // << " including memory transfers).\n";

  } catch (std::runtime_error const &err) {
    std::cerr << "Execution failed with error: \"" << err.what() << "\"."
              << std::endl;
    return 1;
  }

  std::cout << "Verifying results..." << std::flush;
  constexpr int kPrintBodies = 20;
  for (int i = 0; i < kNBodies; ++i) {
    if (i < kPrintBodies) {
      std::cout << position[i] << " / " << positionRef[i] << ", "
                << velocity[i] << " / " << velocityRef[i] << "\n";
    }
    for (int d = 0; d < kDims; ++d) {
      const auto diff = std::abs(position[i][d] - positionHardware[i][d]);
      if (diff >= 1e-4) {
        std::cerr << "Mismatch in hardware implementation at index " << i
                  << ": " << positionHardware[i] << " (should be "
                  << position[i] << ")." << std::endl;
        return 1;
      }
    }
  }
  std::cout << " Done." << std::endl;

  return 0;
}
