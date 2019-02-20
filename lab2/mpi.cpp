#include <mpi.h>
#include <cstring>
#include <stdlib.h>
#include "../lab1/gemm.h"

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  int rank;
  int size;
  const int kRoot = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  float* aFlat = (float*) aligned_alloc(64, kI * kK * sizeof(float));
  float* bFlat = (float*) aligned_alloc(64, kK * kJ * sizeof(float));
  float* cTemp = (float*) aligned_alloc(64, kI * kJ * sizeof(float));
  for(int i = 0; i < kI * kJ; ++i) {
    cTemp[i] = 0;
  }
  if (rank == kRoot) {
    for (int i = 0; i < kI; ++i) {
      std::memset(c[i], 0, sizeof(float) * kJ);
    }
    for (int i = 0; i < kI * kK; ++i) {
      int row = i / kK;
      int col = i % kK;
      aFlat[row * kK + col] = a[row][col];
    }
    for (int i = 0; i < kK * kJ; ++i) {
      int row = i / kJ;
      int col = i % kJ;
      bFlat[row * kJ + col] = b[row][col];
    }
  }
  MPI_Bcast(aFlat, kI * kK, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
  MPI_Bcast(bFlat, kK * kJ, MPI_FLOAT, kRoot, MPI_COMM_WORLD);
  int nI = 64;
  int nJ = 1024;
  int nK = 8;
  int iBlocks = (kI + nI - 1)/nI;
  int jBlocks = (kJ + nJ - 1)/nJ;
  int kBlocks = (kK + nK - 1)/nK;
  int stride = (iBlocks + size - 1)/size;
  for (int ii = rank * stride; ii < (rank + 1) * stride && ii < iBlocks; ++ii) {
    for (int jj = 0; jj < jBlocks; ++jj) {
      for (int kk = 0; kk < kBlocks; ++kk) {
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
  MPI_Reduce(cTemp, c, kI * kJ, MPI_FLOAT, MPI_SUM, kRoot, MPI_COMM_WORLD);
  free(cTemp);
  free(bFlat);
  free(aFlat);
}
