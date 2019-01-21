// Header inclusions, if any...

#include <cstring>
#include <stdlib.h>
#include <math.h>

#include "gemm.h"
#include "omp.h"


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
			 float c[kI][kJ]) {
  int n = 1024;
#pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  float* bT = (float*) malloc(kJ * kK * sizeof(float));
#pragma omp parallel for
  for (int i = 0; i < kK * kJ; ++i) {
    int row = i / kJ;
    int col = i % kJ;
    bT[col * kK + row] = b[row][col];
  }

#pragma omp parallel
  {
    float* cTemp = (float*) malloc(kI * kJ * sizeof(float));
    int iBlocks = (kI + n - 1)/n;
    int jBlocks = (kJ + n - 1)/n;
    int kBlocks = (kK + n - 1)/n;
#pragma omp for schedule(static, kBlocks)
    for (int block = 0; block < iBlocks * jBlocks * kBlocks; ++block) {
      int ii = block / (kBlocks * jBlocks);
      int jj = (block / kBlocks) % jBlocks;
      int kk = block % kBlocks;
      for (int i = ii * n; i < ii * n + n && i < kI; ++i) {
        for (int j = jj * n; j < jj * n + n && j < kJ; ++j) {
          for (int k = kk * n; k < kk * n + n && k < kK; ++k) {
            cTemp[i*kJ+j] += a[i][k] * bT[j*kK+k];
          }
        }
      }
    }
#pragma omp critical
    {
      for (int i=0; i<kI; ++i) {
        for (int j=0; j<kJ; ++j) {
          c[i][j]+=cTemp[i*kJ+j];
	}
      }
    }
    free(cTemp);
  }
}
