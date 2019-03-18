const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

#define max(a, b) ((a) > (b) ? (a) : (b))
#define input(j, h, w) input[j][h][w]
#define output(i, h, w) output[i][h][w]
#define weight(i, j, p, q) weight[i][j][p][q]

#pragma ACCEL kernel
void CnnKernel(const float input[kNum][kInImSize][kInImSize],
               const float weight[kNum][kNum][kKernel][kKernel], const float bias[kNum],
               float output[kNum][kOutImSize][kOutImSize]) {
  float output_buf[kImSize][kImSize];
  float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel];
  float weight_buf[kKernel][kKernel];

  for (int i = 0; i < kNum; i++) {
    for (int h = 0; h < kImSize; h++) {
        for (int w = 0; w < kImSize; w++) {
          output_buf[h][w] = bias[i];
        }
    }
    for (int j = 0; j < kNum; j++) {
#pragma ACCEL pipeline flatten
      for (int p = 0; p < kKernel; p++) {
        for (int q = 0; q < kKernel; q++) {
          weight_buf[p][q] = weight(i, j, p, q);
        }
      }
      for (int h = 0; h < kInImSize; h++) {
#pragma ACCEL pipeline flatten
#pragma ACCEL parallel factor=4
        for (int w = 0; w < kInImSize; w++) {
          for (int q = 0; q < kKernel; q++) {
            input_buf[h][w - q + kKernel - 1][q] = input(j, h, w);
          }
        }
      }
      for (int h = 0; h < kImSize; h++) {
#pragma ACCEL pipeline flatten
#pragma ACCEL parallel factor=8
#pragma ACCEL false_dependence variable=output_buf
        for (int w = 0; w < kImSize; w++) {
          float tmp = 0;
          for (int p = 0; p < kKernel; p++) {
            for (int q = 0; q < kKernel; q++) {
              tmp +=
                weight_buf[p][q] * input_buf[h + p][w + kKernel - 1][q];
            }
          }
          output_buf[h][w] += tmp;
        }
      }
    }
    for (int h = 0; h < kOutImSize; h++) {
      for (int w = 0; w < kOutImSize; w++) {
        output(i, h, w) =  max(0, max(
                                      max(output_buf[h * 2][w * 2    ], output_buf[h * 2 + 1][w * 2    ]),
                                      max(output_buf[h * 2][w * 2 + 1], output_buf[h * 2 + 1][w * 2 + 1])));
      }
    }
  }
}
