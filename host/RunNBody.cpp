/// @author    Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
/// @date      January 2018
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "NBody.h"
#include "hlslib/xilinx/SDAccel.h"

#include <iostream>
#include <fstream>
#include <random>

std::vector<MemoryPack_t> PackVelocity(std::vector<Vec_t> const &velocity) {
  std::cout << "Packing velocity vector..." << std::flush;
  std::vector<MemoryPack_t> velocityMem(2 * kMemorySizeVelocity);
  for (int i = 0; i < kDims * kNBodies; ++i) {
    velocityMem[i / kMemoryWidth][i % kMemoryWidth] =
        velocity[i / kDims][i % kDims];
  }
  std::cout << " Done." << std::endl;
  return velocityMem;
}

std::vector<MemoryPack_t> PackPosition(std::vector<PosMass_t> const &position) {
  std::cout << "Packing position vector..." << std::flush;
  std::vector<MemoryPack_t> positionMem(2 * kMemorySizePosition);
  for (int i = 0; i < (kDims + 1) * kNBodies; ++i) {
    positionMem[i / kMemoryWidth][i % kMemoryWidth] =
        position[i / (kDims + 1)][i % (kDims + 1)];
  }
  std::cout << " Done." << std::endl;
  return positionMem;
}

std::vector<Vec_t> UnpackVelocity(
    std::vector<MemoryPack_t> const &velocityMem) {
  std::cout << "Unpacking velocity vector..." << std::flush;
  std::vector<Vec_t> velocity(kDims * kNBodies);
  for (int i = 0; i < kDims * kNBodies; ++i) {
    velocity[i / kDims][i % kDims] =
        velocityMem[i / kMemoryWidth][i % kMemoryWidth];
  }
  std::cout << " Done." << std::endl;
  return velocity;
}

std::vector<PosMass_t> UnpackPosition(
    std::vector<MemoryPack_t> const &positionMem) {
  std::cout << "Unpacking position vector..." << std::flush;
  std::vector<PosMass_t> position((kDims + 1) * kNBodies);
  for (int i = 0; i < 2 * (kDims + 1) * kNBodies; ++i) {
    position[i / (kDims + 1)][i % (kDims + 1)] =
        positionMem[i / kMemoryWidth][i % kMemoryWidth];
  }
  std::cout << " Done." << std::endl;
  return position;
}

void RunSoftwareEmulation(std::vector<PosMass_t> &position,
                          std::vector<Vec_t> &velocity, unsigned timesteps) {

  auto positionMem = PackPosition(position);
  auto velocityMem = PackVelocity(velocity);

  hlslib::Stream<MemoryPack_t> velocityReadMemory("velocityReadMemory");
  hlslib::Stream<MemoryPack_t> velocityWriteMemory("velocityWriteMemory");
  hlslib::Stream<Vec_t> velocityReadKernel("velocityReadKernel");
  hlslib::Stream<Vec_t> velocityWriteKernel("velocityWriteKernel");
  hlslib::Stream<MemoryPack_t> positionMassReadMemory("positionMassReadMemory");
  hlslib::Stream<MemoryPack_t> positionMassWriteMemory(
      "positionMassWriteMemory");
  hlslib::Stream<PosMass_t> positionMassReadKernel("positionMassReadKernel");
  hlslib::Stream<PosMass_t> positionMassWriteKernel("positionMassWriteKernel");

  std::cout << "Running emulation of hardware implementation..." << std::flush;
  NBody(timesteps, &positionMem[0], &positionMem[0], &velocityMem[0],
        &velocityMem[0], velocityReadMemory, velocityReadKernel,
        positionMassReadMemory, positionMassReadKernel, velocityWriteKernel,
        velocityWriteMemory, positionMassWriteKernel, positionMassWriteMemory);
  std::cout << " Done.\n";

  position = UnpackPosition(positionMem);
  velocity = UnpackVelocity(velocityMem);
}

int main(int argc, char **argv) {

  if (argc > 2) {
    std::cerr << "Usage: ./RunNBody.exe [<number of timesteps>]" << std::endl;
    return 1;
  }

  const int timesteps = (argc == 2) ? std::stoi(argv[1]) : kSteps;
  std::cout << "Running for " << timesteps << " timesteps.\n";

  std::vector<Vec_t> velocity(kNBodies);
  std::vector<PosMass_t> position(2 * kNBodies); // Double buffering

  Data_t clusterScale = 1.54f;
  Data_t velocityScale = 8.0f;

  std::random_device rd;
  std::default_random_engine rng(5); // Use fixed seed
  std::uniform_real_distribution<double> distMass(1e10, 1e12);
  std::uniform_real_distribution<double> distVelocity(-1e3, 1e3);
  std::uniform_real_distribution<double> distPosition(-1e6, 1e6);

  Data_t scale = clusterScale * std::max<Data_t>(1.0, kNBodies / (1024.0));
  Data_t vscale = velocityScale * scale;

  //The number of bodies used as padding to make sure all the writes get pushed through.
  //BE CAREFUL: DO NOT USE THE LAST flushfactor ELEMENTS
  constexpr int flushfactor = 256;
  static_assert(kNBodies > flushfactor, "Number of bodies too low");

  int i = 0;

  std::cout << "Filling random values..." << std::flush;
  while (i < kNBodies) {
    Vec_t point;
    Data_t lenSqr = 0.0;
    for (int j = 0; j < kDims; j++) {
      point[j] = rand() / Data_t(RAND_MAX) * 2 - 1;
      lenSqr += point[j] * point[j];
    }
    if (lenSqr > 1) continue;

    Vec_t vel;
    lenSqr = 0.0;
    for (int j = 0; j < kDims; j++) {
      vel[j] = rand() / Data_t(RAND_MAX) * 2 - 1;
      lenSqr += vel[j] * vel[j];
    }
    if (lenSqr > 1) continue;

    for (int j = 0; j < kDims; j++) {
      position[i][j] = point[j] * scale;
      velocity[i][j] = vel[j] * vscale;
    }

    //set mass to 0 for the padding
    if(i < kNBodies - flushfactor){
      position[i][kDims] = 1.0f;  // mass
    }else{
      position[i][kDims] = 0.0f;
    }
    i++;
  }
  std::cout << "Done." << std::endl;

  std::vector<PosMass_t> positionHardware(position);
  std::vector<Vec_t> velocityHardware(velocity);

  auto positionMem = PackPosition(position);
  auto velocityMem = PackVelocity(velocity);
  double timeelapsed;
  try {

    std::cout << "Initializing OpenCL context..." << std::flush;
    hlslib::ocl::Context context;
    std::cout << " Done.\n";

    std::cout << "Programming device..." << std::flush;
    auto program = context.MakeProgram("NBody.xclbin");
    std::cout << " Done.\n";

    std::cout << "Initializing device memory..." << std::flush;
    auto positionDevice =
        context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::readWrite>(
            hlslib::ocl::MemoryBank::bank0, positionMem.size());
    auto velocityDevice =
        context.MakeBuffer<MemoryPack_t, hlslib::ocl::Access::readWrite>(
            hlslib::ocl::MemoryBank::bank1, velocityMem.size());
    std::cout << " Done.\n";

    auto startWithMemory = std::chrono::high_resolution_clock::now();
    std::cout << "Copying data to device..." << std::flush;
    positionDevice.CopyFromHost(positionMem.cbegin());
    velocityDevice.CopyFromHost(velocityMem.cbegin());
    std::cout << " Done.\n";

    std::cout << "Initializing kernel..." << std::flush;
    auto kernel =
        program.MakeKernel("nbody_kernel", timesteps, positionDevice,
                           positionDevice, velocityDevice, velocityDevice);
    std::cout << " Done.\n";

    std::cout << "Executing kernel..." << std::flush;
    const auto elapsed = kernel.ExecuteTask();
    std::cout << " Done.\n";

    // const auto perf = 1e-9 *
    //                   (2 * static_cast<float>(kSizeN) * kSizeM * kSizeP) /
    //                   elapsed.first;

    std::cout << "Copying back result..." << std::flush;
    positionDevice.CopyToHost(&positionMem[0]);
    velocityDevice.CopyToHost(&velocityMem[0]);
    std::cout << " Done.\n";
    positionHardware = UnpackPosition(positionMem);
    velocityHardware = UnpackVelocity(velocityMem);

    auto endWithMemory = std::chrono::high_resolution_clock::now();
    auto elapsedWithMemory = (endWithMemory - startWithMemory).count();
    timeelapsed = elapsed.first;
    std::cout << "Kernel executed in " << timeelapsed << " seconds.\n";
    // corresponding to a performance of " << perf
    // << " GOp/s (" << perfWithMemory
    // << " including memory transfers).\n";

  } catch (std::runtime_error const &err) {
    std::cerr << "Execution failed with error: \"" << err.what() << "\"."
              << std::endl;
    return 1;
  }
  int fromFile = 1;
  std::vector<PosMass_t> positionRef(position);
  std::vector<Vec_t> velocityRef(velocity);

  std::stringstream filenamestr;
  filenamestr << "../golden/n" << kNBodies << "_t" << timesteps << ".txt";
  auto inFile = std::ifstream(filenamestr.str(), std::ifstream::in);
  if (inFile.good()) {
    std::cout << "Reading from file ... \n";
    fromFile = 0;
    std::string line;
    for(int i = 0; i < kNBodies; i++){
      for(int j = 0; j <= kDims; j ++){
        std::getline(inFile, line);
        positionRef[i][j] = std::atof(line.c_str());
      }
    }
    for(int i = 0; i < kNBodies; i++){
      for(int j = 0; j < kDims; j ++){
        std::getline(inFile, line);
        velocityRef[i][j] = std::atof(line.c_str());
      }
    }
  } else {
    std::cout << "File not found, Computing ... \n";

    RunSoftwareEmulation(positionRef, velocityRef, timesteps);
    std::cout << " Done.\n";

    std::ofstream outputFile;
    outputFile.precision(20);
    outputFile.open(filenamestr.str());
    for(int i = 0; i < kNBodies; i++){
      for(int j = 0; j <= kDims; j ++){
        Data_t val = positionRef[i][j]; 
        outputFile << float(val) << std::endl;
      }
    }
    for(int i = 0; i < kNBodies; i++){
      for(int j = 0; j < kDims; j ++){
        const Data_t val = velocityRef[i][j];
        outputFile << float(val) << std::endl;
      }
    }
    outputFile.close();

    std::cout << "Done writing. \n";
  }

  std::cout << "Verifying results..." << std::endl;
  constexpr int kPrintBodies = 20;
  double totalDiff = 0;
  unsigned mismatches = 0;
  const unsigned offset = (timesteps % 2 == 0) ? 0 : kNBodies;
  for (int i = 0; i < kNBodies; ++i) {
   if (i < kPrintBodies) {
     std::cout << positionHardware[offset + i] << " / "
               << positionRef[offset*fromFile + i] << ", "
               << velocityHardware[i] << " / "
               << velocityRef[i] << "\n";
   }
    bool mismatch = false;
    for (int d = 0; d < kDims; ++d) {
      const Data_t hw = positionHardware[offset + i][d];
      const Data_t ref = positionRef[offset + i][d];
      const auto diff = std::abs(hw - ref);
      totalDiff += diff;
      if (diff >= 1e-4) {
        mismatch = true;
        // std::cerr << "Mismatch in hardware implementation at index " << i
        //           << ": " << positionHardware[i] << " (should be "
        //           << positionRef[i] << ")." << std::endl;
        // return 1;
      }
    }
      // if(mismatch){
      // std::cout << i << " / " << positionHardware[offset + i] << " / "
      // << positionRef[offset + i] << ", "
      // << velocityHardware[i] << " / "
      // << velocityRef[i] << "\n";
      // }
    mismatches += mismatch;
  }
  std::cout << "Mismatches: " << mismatches << " / " << kNBodies << "\n";
  std::cout << "Total difference: " << totalDiff << std::endl;
  if (totalDiff >= kNBodies * 1e-4) {
    std::cerr << "Verification failed." << std::endl;
    return 1;
  }

  //the performance for the not padded bodies
  double performance = (kNBodies - flushfactor)*(kNBodies - flushfactor)*timesteps*(6.*kDims + 4.)/timeelapsed;
  std::cout << "Performance (flop/s): " << performance << "\n";
  std::cout << "Done." << std::endl;

  return 0;
}
