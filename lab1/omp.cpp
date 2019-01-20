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
  float* bT = (float*) malloc(kK*kJ*sizeof(float));
#pragma omp parallel for
  for (int i = 0; i< kK*kJ; ++i) {
    int row = i/kJ;
    int col = i%kJ;
    bT[col*kK+row] = b[row][col];
  }
#pragma omp parallel for
  for (int i = 0; i < kI*kJ; ++i) {
    for (int k = 0; k < kK; ++k) {
      int row = i/kJ;
      int col = i%kJ;
      c[row][col] += a[row][k] * bT[col*kK+k];
    }
  }
  free(bT);
}
