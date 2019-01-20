#include <cstring>

#include <chrono>
#include <iostream>
#include <random>

#include "gemm.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;

void GemmBaseline(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]);
void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]);
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]);

void GemmSequential(const float a[kI][kK], const float b[kK][kJ],
                    float c[kI][kJ]) {
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  for (int i = 0; i < kI; ++i) {
    for (int j = 0; j < kJ; ++j) {
      for (int k = 0; k < kK; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

int Diff(const float c1[kI][kJ], const float c2[kI][kJ]) {
  double diff = 0.;
  auto square = [](float x) -> float { return x * x; };
  for (int i = 0; i < kI; ++i) {
    for (int j = 0; j < kJ; ++j) {
      diff += square(float(c1[i][j]) - c2[i][j]);
    }
  }
  diff /= kI * kJ;
  if (diff > 1e-7f) {
    clog << "Diff: " << diff << endl;
    return 1;
  }
  return 0;
}

void Benchmark(
    void (*gemm)(const float[kI][kK], const float[kK][kJ], float[kI][kJ]),
    const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  const auto begin = steady_clock::now();
  (*gemm)(a, b, c);
  const auto end = steady_clock::now();
  uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  float gflops = 2.0 * kI * kJ * kK / (run_time_us * 1e3);
  clog << "Time: " << run_time_us * 1e-6 << " s\n";
  clog << "Perf: " << gflops << " GFlops\n";
}

int main(int argc, char** argv) {
  // Allocate memory on heap to avoid stack overflow.
  static float a[kI][kK];
  static float b[kK][kJ];
  static float c_base[kI][kJ];
  static float c[kI][kJ];

  bool sequential = false;
  bool parallel = false;
  bool parallel_blocked = false;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "sequential") == 0) {
      sequential = true;
    }
    if (strcmp(argv[i], "parallel") == 0) {
      parallel = true;
    }
    if (strcmp(argv[i], "parallel-blocked") == 0) {
      parallel_blocked = true;
    }
  }

  clog << "Problem size: " << kI << " x " << kK << " x " <<  kJ << endl;

  std::default_random_engine generator(
      steady_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> distribution(0.f, 1.f);

  clog << "Initialize matrices a and b\n";
  for (int i=0; i < kI; ++i) {
    for (int k=0; k < kK; ++k) {
      a[i][k] = distribution(generator);
    }
  }
  for (int k=0; k < kK; ++k) {
    for (int j=0; j < kJ; ++j) {
      b[k][j] = distribution(generator);
    }
  }

  GemmBaseline(a, b, c_base);

  if (sequential) {
    clog << "\nRun sequential GEMM with OpenMP\n";
    Benchmark(&GemmSequential, a, b, c);
    if (Diff(c_base, c) != 0) {
      clog << "Baseline failed\n";
      return 2;
    }
  }

  bool fail = false;
  if (parallel) {
    clog << "\nRun parallel GEMM with OpenMP\n";
    Benchmark(&GemmParallel, a, b, c);
    if (Diff(c_base, c) != 0) {
      fail = true;
    }
  }

  if (parallel_blocked) {
    clog << "\nRun blocked parallel GEMM with OpenMP\n";
    Benchmark(&GemmParallelBlocked, a, b, c);
    if (Diff(c_base, c) != 0) {
      fail = true;
    }
  }

  if (fail) {
    return 1;
  }
  return 0;
}
