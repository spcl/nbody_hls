#include <vector>
#include "Memory.h"

struct NonDivisibleType {
  float a{1};
  float b{2};
  float c{3};
};

constexpr int kIterationsKernel = 96;
constexpr int kIterationsMemory = (3 * kIterationsKernel) / 16;

bool TestUnpack() {
  std::vector<NonDivisibleType> in(kIterationsKernel);
  hlslib::Stream<MemoryPack_t> streamIn(std::numeric_limits<int>::max());
  hlslib::Stream<NonDivisibleType> streamOut(std::numeric_limits<int>::max());
  for (int i = 0; i < kIterationsMemory; ++i) {
    streamIn.WriteOptimistic(
        *(reinterpret_cast<MemoryPack_t const *>(&in[0]) + i));
  }
  UnpackNonDivisible<kIterationsKernel, NonDivisibleType>(streamIn, streamOut);
  for (int i = 0; i < kIterationsKernel; ++i) {
    const auto out = streamOut.ReadOptimistic();
    if (out.a != 1 || out.b != 2 || out.c != 3) {
      std::cout << "Unpack: Mismatch at iteration " << i << ": " << out.a
                << ", " << out.b << ", " << out.c << std::endl;
      return false;
    }
  }
  return true;
}

bool TestPack() {
  std::vector<NonDivisibleType> in(kIterationsKernel);
  std::vector<NonDivisibleType> out(kIterationsKernel);
  hlslib::Stream<NonDivisibleType> streamIn(std::numeric_limits<int>::max());
  hlslib::Stream<MemoryPack_t> streamOut(std::numeric_limits<int>::max());
  for (int i = 0; i < kIterationsKernel; ++i) {
    streamIn.WriteOptimistic(in[i]);
  }
  PackNonDivisible<kIterationsKernel, NonDivisibleType>(streamIn, streamOut);
  for (int i = 0; i < kIterationsMemory; ++i) {
    *(reinterpret_cast<MemoryPack_t *>(&out[0]) + i) =
        streamOut.ReadOptimistic();
  }
  for (int i = 0; i < kIterationsKernel; ++i) {
    const auto val = out[i];
    if (val.a != 1 || val.b != 2 || val.c != 3) {
      std::cout << "Pack: Mismatch at iteration " << i << ": " << val.a << ", "
                << val.b << ", " << val.c << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  if (!TestUnpack()) {
    return 1;
  }
  if (!TestPack()) {
    return 2;
  }
  return 0;
}
