// Header inclusions, if any...

#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "gemm.h"
#include "omp.h"


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  int n = 256;
#pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  float* bT = (float*) malloc(kJ * kK * sizeof(float));
  float* aFlat = (float*) malloc(kI * kK * sizeof(float));
#pragma omp parallel for
  for (int i = 0; i < kK * kJ; ++i) {
    int row = i / kJ;
    int col = i % kJ;
    bT[col * kK + row] = b[row][col];
  }
#pragma omp parallel for
  for (int i = 0; i < kI * kK; ++i) {
    int row = i / kK;
    int col = i % kK;
    aFlat[row * kK + col] = a[row][col];
  }
#pragma omp parallel
  {
    float* cTemp = (float*) malloc(kI * kJ * sizeof(float));
    int iBlocks = (kI + n - 1)/n;
    int jBlocks = (kJ + n - 1)/n;
    int kBlocks = (kK + n - 1)/n;
#pragma omp for
    for (int block = 0; block < iBlocks * jBlocks * kBlocks; ++block) {
      int ii = block / (kBlocks * jBlocks);
      int jj = (block / kBlocks) % jBlocks;
      int kk = block % kBlocks;
      for (int i = ii * n; i < ii * n + n && i < kI; ++i) {
        for (int j = jj * n; j < jj * n + n && j < kJ; ++j) {
          for (int k = kk * n; k < kk * n + n && k < (kK / 4) * 4; k+=4) {
            cTemp[i * kJ + j] += aFlat[i * kK + k] * bT[j * kK + k]
              + aFlat[i * kK + k + 1] * bT[j * kK + k + 1]
              + aFlat[i * kK + k + 2] * bT[j * kK + k + 2]
              + aFlat[i * kK + k + 3] * bT[j * kK + k + 3];
          }
          if (kk * n + n >= (kK / 4) * 4) {
            if (kK % 4 == 1) {
              cTemp[i * kJ + j] += aFlat[i * kK + kK - 1] * bT[j * kK + kK - 1];
            } else if (kK % 4 == 2) {
              cTemp[i * kJ + j] += aFlat[i * kK + kK - 1] * bT[j * kK + kK - 1]
                + aFlat[i * kK + kK - 2] * bT[j * kK + kK - 2];
            } else if (kK % 4 == 3) {
              cTemp[i * kJ + j] += aFlat[i * kK + kK - 1] * bT[j * kK + kK - 1]
                + aFlat[i * kK + kK - 2] * bT[j * kK + kK - 2]
                + aFlat[i * kK + kK - 3] * bT[j * kK + kK - 3];
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
  free(bT);
  free(aFlat);
}
