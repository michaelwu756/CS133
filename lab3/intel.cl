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
  int w = get_global_id(2);

  float C[4];
  C[0] = bias[i];
  C[1] = bias[i];
  C[2] = bias[i];
  C[3] = bias[i];

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
        C[0] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q    ];
        C[1] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q    ];
        C[2] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p    ) * in + w * 2 + q + 1];
        C[3] += weight[i * nk2 + j * k2 + p * k + q] * input[j * in2 + (h * 2 + p + 1) * in + w * 2 + q + 1];
      }
    }
  }

  output[i * o2 + h * o + w] = max(0.f, max(
                                            max(C[0], C[1]),
                                            max(C[2], C[3])));
}
