__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel
void CnnKernel(__global const float* input, __global const float* weight,
               __global const float* bias, __global float* output) {
  int i = get_global_id(0);
  int h = get_global_id(1);
  int w = get_global_id(2) * 8;

  float C[32];
  C[0] = bias[i];
  C[1] = bias[i];
  C[2] = bias[i];
  C[3] = bias[i];
  C[4] = bias[i];
  C[5] = bias[i];
  C[6] = bias[i];
  C[7] = bias[i];
  C[8] = bias[i];
  C[9] = bias[i];
  C[10] = bias[i];
  C[11] = bias[i];
  C[12] = bias[i];
  C[13] = bias[i];
  C[14] = bias[i];
  C[15] = bias[i];
  C[16] = bias[i];
  C[17] = bias[i];
  C[18] = bias[i];
  C[19] = bias[i];
  C[20] = bias[i];
  C[21] = bias[i];
  C[22] = bias[i];
  C[23] = bias[i];
  C[24] = bias[i];
  C[25] = bias[i];
  C[26] = bias[i];
  C[27] = bias[i];
  C[28] = bias[i];
  C[29] = bias[i];
  C[30] = bias[i];
  C[31] = bias[i];

  int nk2 = kNum * kKernel * kKernel;
  int k2 = kKernel * kKernel;
  int k = kKernel;
  int in2 = kInImSize * kInImSize;
  int in = kInImSize;
  int o2 = kOutImSize * kOutImSize;
  int o = kOutImSize;

  for (int j = 0; j < kNum; ++j) {
    for (int p = 0; p < kKernel; ++p) {
      for (int q = 0; q < kKernel; ++q) {
        C[0 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q    ];
        C[1 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 1];
        C[2 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 2];
        C[3 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 3];
        C[4 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 4];
        C[5 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 5];
        C[6 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 6];
        C[7 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 7];
        C[8 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 8];
        C[9 ] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 9];
        C[10] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 10];
        C[11] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 11];
        C[12] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 12];
        C[13] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 13];
        C[14] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 14];
        C[15] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 15];
        C[16] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q    ];
        C[17] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 1];
        C[18] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 2];
        C[19] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 3];
        C[20] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 4];
        C[21] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 5];
        C[22] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 6];
        C[23] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 7];
        C[24] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 8];
        C[25] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 9];
        C[26] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 10];
        C[27] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 11];
        C[28] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 12];
        C[29] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 13];
        C[30] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 14];
        C[31] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 15];
      }
    }
  }

  output[i * o2 + h * o + w    ] = max(0.f, max(
                                                max(C[0 ], C[1 ]),
                                                max(C[16], C[17])));
  output[i * o2 + h * o + w + 1] = max(0.f, max(
                                                max(C[2 ], C[3 ]),
                                                max(C[18], C[19])));
  output[i * o2 + h * o + w + 2] = max(0.f, max(
                                                max(C[4 ], C[5 ]),
                                                max(C[20], C[21])));
  output[i * o2 + h * o + w + 3] = max(0.f, max(
                                                max(C[6 ], C[7 ]),
                                                max(C[22], C[23])));
  output[i * o2 + h * o + w + 4] = max(0.f, max(
                                                max(C[8 ], C[9 ]),
                                                max(C[24], C[25])));
  output[i * o2 + h * o + w + 5] = max(0.f, max(
                                                max(C[10], C[11]),
                                                max(C[26], C[27])));
  output[i * o2 + h * o + w + 6] = max(0.f, max(
                                                max(C[12], C[13]),
                                                max(C[28], C[29])));
  output[i * o2 + h * o + w + 7] = max(0.f, max(
                                                max(C[14], C[15]),
                                                max(C[30], C[31])));
}
