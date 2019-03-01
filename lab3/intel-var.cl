#define N 1
#define M 1
__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {
  int hh = get_global_id(0) / M;
  int ww = get_global_id(0) % M;

  int nk2 = kNum * kKernel * kKernel;
  int k2 = kKernel * kKernel;
  int k = kKernel;
  int in2 = kInImSize * kInImSize;
  int in = kInImSize;
  int o2 = kOutImSize * kOutImSize;
  int o = kOutImSize;

  float C[16];
  for (int i = 0; i < kNum; ++i) {
    for (int h = hh * kOutImSize / N; h < (hh + 1) * kOutImSize / N; ++h) {
      for (int w = ww * kOutImSize / M; w < (ww + 1) * kOutImSize / M; w += 4) {
        C[ 0] = bias[i];
        C[ 1] = bias[i];
        C[ 2] = bias[i];
        C[ 3] = bias[i];
        C[ 4] = bias[i];
        C[ 5] = bias[i];
        C[ 6] = bias[i];
        C[ 7] = bias[i];
        C[ 8] = bias[i];
        C[ 9] = bias[i];
        C[10] = bias[i];
        C[11] = bias[i];
        C[12] = bias[i];
        C[13] = bias[i];
        C[14] = bias[i];
        C[15] = bias[i];

        for (int j = 0; j < kNum; ++j) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
              C[ 0] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q    ];
              C[ 1] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 1];
              C[ 2] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 2];
              C[ 3] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 3];
              C[ 4] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 4];
              C[ 5] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 5];
              C[ 6] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 6];
              C[ 7] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 7];
              C[ 8] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q    ];
              C[ 9] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 1];
              C[10] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 2];
              C[11] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 3];
              C[12] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 4];
              C[13] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 5];
              C[14] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 6];
              C[15] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 7];
            }
          }
        }

        output[i * o2 + h * o + w    ] = max(0.f, max(
                                                      max(C[ 0], C[ 1]),
                                                      max(C[ 8], C[ 9])));
        output[i * o2 + h * o + w + 1] = max(0.f, max(
                                                      max(C[ 2], C[ 3]),
                                                      max(C[10], C[11])));
        output[i * o2 + h * o + w + 2] = max(0.f, max(
                                                      max(C[ 4], C[ 5]),
                                                      max(C[12], C[13])));
        output[i * o2 + h * o + w + 3] = max(0.f, max(
                                                      max(C[ 6], C[ 7]),
                                                      max(C[14], C[15])));
      }
    }
  }
}
