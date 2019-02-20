// Header inclusions, if any...

#include <cstring>
#include <stdlib.h>
#include <immintrin.h>

#include "gemm.h"
#include "omp.h"


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  int nI = 64;
  int nJ = 1024;
  int nK = 8;
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
  }
#pragma omp parallel for
  for (int i = 0; i < kK * kJ; ++i) {
    int row = i / kJ;
    int col = i % kJ;
    bFlat[row * kJ + col] = b[row][col];
  }
#pragma omp parallel
  {
    float* cTemp = (float*) aligned_alloc(64, kI * kJ * sizeof(float));
    for(int i = 0; i < kI * kJ; ++i) {
      cTemp[i] = 0;
    }
    int iBlocks = (kI + nI - 1)/nI;
    int jBlocks = (kJ + nJ - 1)/nJ;
    int kBlocks = (kK + nK - 1)/nK;
#pragma omp for
    for (int ii = 0; ii < iBlocks; ++ii){
      for (int jj = 0; jj < jBlocks; ++jj){
        for (int kk = 0; kk < kBlocks; ++kk){
          for (int i = 0; i < nI; ++i) {
            int iVal = ii * nI + i;
            for (int k = 0; k < nK; ++k) {
              int kVal = kk * nK + k;
              for (int j = 0; j < nJ; j+=16) {
                int jVal = jj * nJ + j;
                cTemp[iVal * kJ + jVal + 0] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 0];
                cTemp[iVal * kJ + jVal + 1] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 1];
                cTemp[iVal * kJ + jVal + 2] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 2];
                cTemp[iVal * kJ + jVal + 3] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 3];
                cTemp[iVal * kJ + jVal + 4] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 4];
                cTemp[iVal * kJ + jVal + 5] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 5];
                cTemp[iVal * kJ + jVal + 6] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 6];
                cTemp[iVal * kJ + jVal + 7] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 7];
                cTemp[iVal * kJ + jVal + 8] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 8];
                cTemp[iVal * kJ + jVal + 9] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 9];
                cTemp[iVal * kJ + jVal + 10] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 10];
                cTemp[iVal * kJ + jVal + 11] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 11];
                cTemp[iVal * kJ + jVal + 12] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 12];
                cTemp[iVal * kJ + jVal + 13] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 13];
                cTemp[iVal * kJ + jVal + 14] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 14];
                cTemp[iVal * kJ + jVal + 15] += aFlat[iVal * kK + kVal] * bFlat[kVal * kJ + jVal + 15];
              }
            }
          }
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
