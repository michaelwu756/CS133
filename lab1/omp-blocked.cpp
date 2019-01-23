// Header inclusions, if any...

#include <cstring>
#include <stdlib.h>
#include <immintrin.h>

#include "gemm.h"
#include "omp.h"


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  int n = 128;
#pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  float* aFlat = (float*) aligned_alloc(64, kI * kK * sizeof(float));
  float* bFlat = (float*) aligned_alloc(64, kK * kJ * sizeof(float));
#pragma omp parallel for
  for (int i = 0; i < kI * kK; ++i) {
    int row = i / kK;
    int col = i % kK;
    aFlat[row * kK + col] = a[row][col];
    bFlat[row * kJ + col] = b[row][col];
  }
#pragma omp parallel
  {
    float* cTemp = (float*) aligned_alloc(64, kI * kJ * sizeof(float));
    __m512 simd1, simd2, simd3, simd4;
    int iBlocks = (kI + n - 1)/n;
    int jBlocks = (kJ + n - 1)/n;
    int kBlocks = (kK + n - 1)/n;
#pragma omp for
    for (int block = 0; block < iBlocks * jBlocks * kBlocks; ++block) {
      int ii = block / (kBlocks * jBlocks);
      int jj = (block / kBlocks) % jBlocks;
      int kk = block % kBlocks;
      for (int i = 0; i < n; ++i) {
        int iVal = ii * n + i;
        for (int j = 0; j < n; j+=16) {
          int jVal = jj * n + j;
          float* cBase = cTemp + iVal * kJ + jVal;
          simd3 = _mm512_load_ps(cBase);
          for (int k = 0; k < n; ++k) {
            int kVal = kk * n + k;
            float* aBase = aFlat + iVal * kK + kVal;
            float* bBase = bFlat + kVal * kJ + jVal;
            simd1 = _mm512_set1_ps(*aBase);
            simd2 = _mm512_load_ps(bBase);
            simd4 = _mm512_mul_ps(simd1, simd2);
            simd3 = _mm512_add_ps(simd3, simd4);
          }
          _mm512_store_ps(cBase, simd3);
        }
      }
    }
#pragma omp critical
    for (int i = 0; i < kI; ++i) {
      for (int j = 0; j < kJ; ++j) {
        c[i][j] += cTemp[i * kJ + j];
      }
    }
    free(cTemp);
  }
  free(bFlat);
  free(aFlat);
}
