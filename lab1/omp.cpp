// Header inclusions, if any...

#include <cstring>
#include <stdlib.h>

#include "gemm.h"
#include "omp.h"

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
#pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  float* bT = (float*) malloc(kK * kJ * sizeof(float));
#pragma omp parallel for
  for (int i = 0; i< kK * kJ; ++i) {
    int row = i / kJ;
    int col = i % kJ;
    bT[col * kK + row] = b[row][col];
  }
  float* aFlat = (float*) malloc(kI * kK * sizeof(float));
#pragma omp parallel for
  for (int i = 0; i < kI * kK; ++i) {
    int row = i / kK;
    int col = i % kK;
    aFlat[row * kK + col] = a[row][col];
  }
#pragma omp parallel for
  for (int i = 0; i < kI * kJ; ++i) {
    int row = i / kJ;
    int col = i % kJ;
    for (int k = 0; k < (kK / 4) * 4; k+=4) {
      c[row][col] += aFlat[row * kK + k] * bT[col * kK + k]
        + aFlat[row * kK + k + 1] * bT[col * kK + k + 1]
        + aFlat[row * kK + k + 2] * bT[col * kK + k + 2]
        + aFlat[row * kK + k + 3] * bT[col * kK + k + 3];
    }
    if (kK % 4 == 1) {
      c[row][col] += aFlat[row * kK + kK - 1] * bT[col * kK + kK - 1];
    } else if (kK % 4 == 2) {
      c[row][col] += aFlat[row * kK + kK - 1] * bT[col * kK + kK - 1]
        + aFlat[row * kK + kK - 2] * bT[col * kK + kK - 2];
    } else if (kK % 4 == 3) {
      c[row][col] += aFlat[row * kK + kK - 1] * bT[col * kK + kK - 1]
        + aFlat[row * kK + kK - 2] * bT[col * kK + kK - 2]
        + aFlat[row * kK + kK - 3] * bT[col * kK + kK - 3];
    }
  }
  free(bT);
  free(aFlat);
}
