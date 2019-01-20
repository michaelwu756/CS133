// Header inclusions, if any...

#include <cstring>
#include <stdlib.h>
#include <math.h>

#include "gemm.h"
#include "omp.h"


void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
			 float c[kI][kJ]) {
  int n=round(sqrt(double(kN)));
#pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  float* bT = (float*) malloc(kJ*kK*sizeof(float));
#pragma omp parallel for
  for (int i = 0; i< kK*kJ; ++i) {
    int row = i/kJ;
    int col = i%kJ;
    bT[col*kK+row] = b[row][col];
  }

#pragma omp parallel
  {
    float* cTemp = (float*) malloc(kI*kJ*sizeof(float));
#pragma omp for
    for (int ii = 0; ii < kI; ii += n) {
      for (int jj = 0; jj < kJ; jj += n) {
	for (int kk = 0; kk < kK; kk += n) {
	  for (int i = ii; i < ii + n && i < kI; ++i) {
	    for (int j = jj; j < jj + n && j < kJ; ++j) {
	      for (int k = kk; k < kk + n && k < kK; ++k) {
		cTemp[i*kJ+j] += a[i][k] * bT[j*kK+k];
	      }
	    }
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
  // Using declarations, if any...
  /*void GemmParallelBlockedRecursive(float* a, float* b, float* c, int size);

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  if (kK%2!=0 || kK!=kI || kK!=kJ) {
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
  } else {
    float* atemp = (float*) malloc(kK*kK*sizeof(float));
    float* btemp = (float*) malloc(kK*kK*sizeof(float));
    float* ctemp = (float*) malloc(kK*kK*sizeof(float));
    GemmParallelBlockedRecursive(atemp, btemp, ctemp, kK);
    for (int i=0; i<kK*kK; ++i) {
      int row = i/kK;
      int col = i%kK;
      c[row][col] = ctemp[row*kK+col];
    }
  }
}
void GemmParallelBlockedRecursive(float* a, float* b, float* c, int size) {
  if (size%2!=0 || size < sqrt(double(kK))) {
    float* bT = (float*) malloc(size*size*sizeof(float));
    for (int i = 0; i< size*size; ++i) {
      int row = i/size;
      int col = i%size;
      bT[col*size+row] = b[row*size+col];
    }
#pragma omp parallel for
    for (int i = 0; i < size*size; ++i) {
      for (int k = 0; k < size; ++k) {
	int row = i/size;
	int col = i%size;
	c[row*size+col] += a[row*size+k] * bT[col*size+k];
      }
    }
    free(bT);
  } else {
    int n = size/2;
    float* a11 = (float*) malloc(n*n*sizeof(float));
    float* a12 = (float*) malloc(n*n*sizeof(float));
    float* a21 = (float*) malloc(n*n*sizeof(float));
    float* a22 = (float*) malloc(n*n*sizeof(float));
    float* b11 = (float*) malloc(n*n*sizeof(float));
    float* b12 = (float*) malloc(n*n*sizeof(float));
    float* b21 = (float*) malloc(n*n*sizeof(float));
    float* b22 = (float*) malloc(n*n*sizeof(float));
    float* c11x11 = (float*) malloc(n*n*sizeof(float));
    float* c11x12 = (float*) malloc(n*n*sizeof(float));
    float* c12x21 = (float*) malloc(n*n*sizeof(float));
    float* c12x22 = (float*) malloc(n*n*sizeof(float));
    float* c21x11 = (float*) malloc(n*n*sizeof(float));
    float* c21x12 = (float*) malloc(n*n*sizeof(float));
    float* c22x21 = (float*) malloc(n*n*sizeof(float));
    float* c22x22 = (float*) malloc(n*n*sizeof(float));
    for (int i = 0; i<n*n; ++i) {
      int row = i/n;
      int col = i%n;
      a11[i] = a[row*n+col];
      a12[i] = a[row*n+n+col];
      a21[i] = a[(n+row)*n+col];
      a22[i] = a[(n+row)*n+col];
      b11[i] = b[row*n+col];
      b12[i] = b[row*n+n+col];
      b21[i] = b[(n+row)*n+col];
      b22[i] = b[(n+row)*n+n+col];
    }
#pragma omp parallel
    {
#pragma omp task
      GemmParallelBlockedRecursive(a11, b11, c11x11, n);
#pragma omp task
      GemmParallelBlockedRecursive(a11, b12, c11x12, n);
#pragma omp task
      GemmParallelBlockedRecursive(a12, b21, c12x21, n);
#pragma omp task
      GemmParallelBlockedRecursive(a12, b22, c12x22, n);
#pragma omp task
      GemmParallelBlockedRecursive(a21, b11, c21x11, n);
#pragma omp task
      GemmParallelBlockedRecursive(a21, b12, c21x12, n);
#pragma omp task
      GemmParallelBlockedRecursive(a22, b21, c22x21, n);
#pragma omp task
      GemmParallelBlockedRecursive(a22, b22, c22x22, n);
    }
    for (int i = 0; i<n*n; ++i) {
      int row = i/n;
      int col = i%n;
      c[row*n+col] = c11x11[row*n+col] + c12x21[row*n+col];
      c[row*n+n+col] = c11x12[row*n+col] + c12x22[row*n+col];
      c[(n+row)*n+col] = c21x11[row*n+col] + c22x21[row*n+col];
      c[(n+row)*n+n+col] = c21x12[row*n+col] + c22x22[row*n+col];
    }
    free(a11);
    free(a12);
    free(a21);
    free(a22);
    free(b11);
    free(b12);
    free(b21);
    free(b22);
    free(c11x11);
    free(c11x12);
    free(c12x21);
    free(c12x22);
    free(c21x11);
    free(c21x12);
    free(c22x21);
    free(c22x22);
  }
}


*/
