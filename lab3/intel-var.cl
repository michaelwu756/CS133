#define N 1
__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {
  int ii = get_global_id(0);

  int nk2 = kNum * kKernel * kKernel;
  int k2 = kKernel * kKernel;
  int k = kKernel;
  int in2 = kInImSize * kInImSize;
  int in = kInImSize;
  int o2 = kOutImSize * kOutImSize;
  int o = kOutImSize;

  static float C[kNum][kImSize][kImSize];
  for (int i = ii * kNum / N; i < (ii + 1) * kNum / N; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; h < kOutImSize / 4; ++w) {
        C[i][h    ][w    ] = bias[i];
        C[i][h    ][w + 1] = bias[i];
        C[i][h    ][w + 2] = bias[i];
        C[i][h    ][w + 3] = bias[i];
        C[i][h    ][w + 4] = bias[i];
        C[i][h    ][w + 5] = bias[i];
        C[i][h    ][w + 6] = bias[i];
        C[i][h    ][w + 7] = bias[i];
        C[i][h + 1][w    ] = bias[i];
        C[i][h + 1][w + 1] = bias[i];
        C[i][h + 1][w + 2] = bias[i];
        C[i][h + 1][w + 3] = bias[i];
        C[i][h + 1][w + 4] = bias[i];
        C[i][h + 1][w + 5] = bias[i];
        C[i][h + 1][w + 6] = bias[i];
        C[i][h + 1][w + 7] = bias[i];

        for (int j = 0; j < kNum; ++j) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
              C[i][h    ][w    ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q    ];
              C[i][h    ][w + 1] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 1];
              C[i][h    ][w + 2] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 2];
              C[i][h    ][w + 3] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 3];
              C[i][h    ][w + 4] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 4];
              C[i][h    ][w + 5] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 5];
              C[i][h    ][w + 6] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 6];
              C[i][h    ][w + 7] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 7];
              C[i][h + 1][w    ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q    ];
              C[i][h + 1][w + 1] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 1];
              C[i][h + 1][w + 2] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 2];
              C[i][h + 1][w + 3] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 3];
              C[i][h + 1][w + 4] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 4];
              C[i][h + 1][w + 5] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 5];
              C[i][h + 1][w + 6] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 6];
              C[i][h + 1][w + 7] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 7];
            }
          }
        }

        output[i * o2 + h * o + w    ] = max(0.f, max(
                                                      max(C[i][h    ][w    ], C[i][h    ][w + 1]),
                                                      max(C[i][h + 1][w    ], C[i][h + 1][w + 1])));
        output[i * o2 + h * o + w + 1] = max(0.f, max(
                                                      max(C[i][h    ][w + 2], C[i][h    ][w + 3]),
                                                      max(C[i][h + 1][w + 2], C[i][h + 1][w + 3])));
        output[i * o2 + h * o + w + 2] = max(0.f, max(
                                                      max(C[i][h    ][w + 4], C[i][h    ][w + 5]),
                                                      max(C[i][h + 1][w + 4], C[i][h + 1][w + 5])));
        output[i * o2 + h * o + w + 3] = max(0.f, max(
                                                      max(C[i][h    ][w + 6], C[i][h    ][w + 7]),
                                                      max(C[i][h + 1][w + 6], C[i][h + 1][w + 7])));
      }
    }
  }
}
