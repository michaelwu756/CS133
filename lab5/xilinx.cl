__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define max(a, b) ((a) > (b) ? (a) : (b))
#define input(j, h, w) input[((j) * kInImSize * kInImSize + (h) * kInImSize + (w))]
#define output(i, h, w) output[((i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w))]
#define weight(i, j, p, q) weight[((i) * kNum * kKernel * kKernel + (j) * kKernel * kKernel + (p) * kKernel + (q))]

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  float output_buf[kImSize][kImSize]
  __attribute__((xcl_array_partition(cyclic, 8, 1)))
  __attribute__((xcl_array_partition(cyclic, 2, 2)))
  ;

  float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
  __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
  __attribute__((xcl_array_partition(cyclic, 2, 2)))  // cyclic partition factor of 2 in dim 2 of input_buf
  __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
  ;

  float weight_buf[kKernel][kKernel] //buffer of weight
  __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
  __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
  ;

  float output_buf1[kImSize][kImSize]
  __attribute__((xcl_array_partition(cyclic, 8, 1)))
  __attribute__((xcl_array_partition(cyclic, 2, 2)))
  ;

  float input_buf1[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
  __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
  __attribute__((xcl_array_partition(cyclic, 2, 2)))  // cyclic partition factor of 2 in dim 2 of input_buf
  __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
  ;

  float weight_buf1[kKernel][kKernel] //buffer of weight
  __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
  __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
  ;

  for (int i = 0; i < kNum; i+=2) {
    //copy bias here
    load_bias:
    for (int h = 0; h < kImSize; h++) {
        for (int w = 0; w < kImSize; w++) {
          output_buf[h][w] = bias[i];
        }
    }
    for (int j = 0; j < kNum; j++) {
      //copy weight here
      load_weight:
      __attribute__((xcl_pipeline_loop))
      for (int p = 0; p < kKernel; p++) {
        for (int q = 0; q < kKernel; q++) {
          weight_buf[p][q] = weight(i, j, p, q);
        }
      }
      for (int h = 0; h < kInImSize; h++) {
        //input load loop
        load_in:
        __attribute__((xcl_pipeline_loop))
        for (int w = 0; w < kInImSize; w+=4) {
          for (int q = 0; q < kKernel; q++) { //make kKernel copy of input(j,h,w)
            input_buf[h][w     - q + kKernel - 1][q] = input(j, h, w    );
            input_buf[h][w + 1 - q + kKernel - 1][q] = input(j, h, w + 1);
            input_buf[h][w + 2 - q + kKernel - 1][q] = input(j, h, w + 2);
            input_buf[h][w + 3 - q + kKernel - 1][q] = input(j, h, w + 3);
          }
        }
      }

      for (int h = 0; h < kImSize; h+=8) {
        //convolution loop
        conv:
        __attribute__((xcl_pipeline_loop))
        for (int w = 0; w < kImSize; w+=2) { //pipelined loop
          float tmp0 = 0;
          float tmp1 = 0;
          float tmp2 = 0;
          float tmp3 = 0;
          float tmp4 = 0;
          float tmp5 = 0;
          float tmp6 = 0;
          float tmp7 = 0;
          float tmp8 = 0;
          float tmp9 = 0;
          float tmp10 = 0;
          float tmp11 = 0;
          float tmp12 = 0;
          float tmp13 = 0;
          float tmp14 = 0;
          float tmp15 = 0;
          for (int p = 0; p < kKernel; p++) {  // unrolled loop
            for (int q = 0; q < kKernel; q++) {  //unrolled loop
              tmp0 += //will be synthesized into tree reduction
                weight_buf[p][q] * input_buf[h     + p][w + kKernel - 1][q];
              tmp1 +=
                weight_buf[p][q] * input_buf[h + 1 + p][w + kKernel - 1][q];
              tmp2 +=
                weight_buf[p][q] * input_buf[h + 2 + p][w + kKernel - 1][q];
              tmp3 +=
                weight_buf[p][q] * input_buf[h + 3 + p][w + kKernel - 1][q];
              tmp4 +=
                weight_buf[p][q] * input_buf[h + 4 + p][w + kKernel - 1][q];
              tmp5 +=
                weight_buf[p][q] * input_buf[h + 5 + p][w + kKernel - 1][q];
              tmp6 +=
                weight_buf[p][q] * input_buf[h + 6 + p][w + kKernel - 1][q];
              tmp7 +=
                weight_buf[p][q] * input_buf[h + 7 + p][w + kKernel - 1][q];
              tmp8 +=
                weight_buf[p][q] * input_buf[h     + p][w + kKernel    ][q];
              tmp9 +=
                weight_buf[p][q] * input_buf[h + 1 + p][w + kKernel    ][q];
              tmp10 +=
                weight_buf[p][q] * input_buf[h + 2 + p][w + kKernel    ][q];
              tmp11 +=
                weight_buf[p][q] * input_buf[h + 3 + p][w + kKernel    ][q];
              tmp12 +=
                weight_buf[p][q] * input_buf[h + 4 + p][w + kKernel    ][q];
              tmp13 +=
                weight_buf[p][q] * input_buf[h + 5 + p][w + kKernel    ][q];
              tmp14 +=
                weight_buf[p][q] * input_buf[h + 6 + p][w + kKernel    ][q];
              tmp15 +=
                weight_buf[p][q] * input_buf[h + 7 + p][w + kKernel    ][q];

            }
          }
          output_buf[h    ][w    ] += tmp0; //store reduction result
          output_buf[h + 1][w    ] += tmp1;
          output_buf[h + 2][w    ] += tmp2;
          output_buf[h + 3][w    ] += tmp3;
          output_buf[h + 4][w    ] += tmp4;
          output_buf[h + 5][w    ] += tmp5;
          output_buf[h + 6][w    ] += tmp6;
          output_buf[h + 7][w    ] += tmp7;
          output_buf[h    ][w + 1] += tmp8;
          output_buf[h + 1][w + 1] += tmp9;
          output_buf[h + 2][w + 1] += tmp10;
          output_buf[h + 3][w + 1] += tmp11;
          output_buf[h + 4][w + 1] += tmp12;
          output_buf[h + 5][w + 1] += tmp13;
          output_buf[h + 6][w + 1] += tmp14;
          output_buf[h + 7][w + 1] += tmp15;
        }
      }
    }
    //copy output here
    store:
    for (int h = 0; h < kOutImSize; h++) {
      for (int w = 0; w < kOutImSize; w++) {
        output(i, h, w) =  max(0, max(
                                      max(output_buf[h * 2][w * 2    ], output_buf[h * 2 + 1][w * 2    ]),
                                      max(output_buf[h * 2][w * 2 + 1], output_buf[h * 2 + 1][w * 2 + 1])));
      }
    }
    //copy bias here
    load_bias1:
    for (int h = 0; h < kImSize; h++) {
        for (int w = 0; w < kImSize; w++) {
          output_buf1[h][w] = bias[i + 1];
        }
    }
    for (int j = 0; j < kNum; j++) {
      //copy weight here
      load_weight1:
      __attribute__((xcl_pipeline_loop))
      for (int p = 0; p < kKernel; p++) {
        for (int q = 0; q < kKernel; q++) {
          weight_buf1[p][q] = weight(i + 1, j, p, q);
        }
      }
      for (int h = 0; h < kInImSize; h++) {
        //input load loop
        load_in1:
        __attribute__((xcl_pipeline_loop))
        for (int w = 0; w < kInImSize; w+=4) {
          for (int q = 0; q < kKernel; q++) { //make kKernel copy of input(j,h,w)
            input_buf1[h][w     - q + kKernel - 1][q] = input(j, h, w    );
            input_buf1[h][w + 1 - q + kKernel - 1][q] = input(j, h, w + 1);
            input_buf1[h][w + 2 - q + kKernel - 1][q] = input(j, h, w + 2);
            input_buf1[h][w + 3 - q + kKernel - 1][q] = input(j, h, w + 3);
          }
        }
      }
      for (int h = 0; h < kImSize; h+=8) {
        //convolution loop
        conv1:
        __attribute__((xcl_pipeline_loop))
        for (int w = 0; w < kImSize; w+=2) { //pipelined loop
          float tmp0 = 0;
          float tmp1 = 0;
          float tmp2 = 0;
          float tmp3 = 0;
          float tmp4 = 0;
          float tmp5 = 0;
          float tmp6 = 0;
          float tmp7 = 0;
          float tmp8 = 0;
          float tmp9 = 0;
          float tmp10 = 0;
          float tmp11 = 0;
          float tmp12 = 0;
          float tmp13 = 0;
          float tmp14 = 0;
          float tmp15 = 0;
          for (int p = 0; p < kKernel; p++) {  // unrolled loop
            for (int q = 0; q < kKernel; q++) {  //unrolled loop
              tmp0 += //will be synthesized into tree reduction
                weight_buf1[p][q] * input_buf1[h     + p][w + kKernel - 1][q];
              tmp1 +=
                weight_buf1[p][q] * input_buf1[h + 1 + p][w + kKernel - 1][q];
              tmp2 +=
                weight_buf1[p][q] * input_buf1[h + 2 + p][w + kKernel - 1][q];
              tmp3 +=
                weight_buf1[p][q] * input_buf1[h + 3 + p][w + kKernel - 1][q];
              tmp4 +=
                weight_buf1[p][q] * input_buf1[h + 4 + p][w + kKernel - 1][q];
              tmp5 +=
                weight_buf1[p][q] * input_buf1[h + 5 + p][w + kKernel - 1][q];
              tmp6 +=
                weight_buf1[p][q] * input_buf1[h + 6 + p][w + kKernel - 1][q];
              tmp7 +=
                weight_buf1[p][q] * input_buf1[h + 7 + p][w + kKernel - 1][q];
              tmp8 +=
                weight_buf1[p][q] * input_buf1[h     + p][w + kKernel    ][q];
              tmp9 +=
                weight_buf1[p][q] * input_buf1[h + 1 + p][w + kKernel    ][q];
              tmp10 +=
                weight_buf1[p][q] * input_buf1[h + 2 + p][w + kKernel    ][q];
              tmp11 +=
                weight_buf1[p][q] * input_buf1[h + 3 + p][w + kKernel    ][q];
              tmp12 +=
                weight_buf1[p][q] * input_buf1[h + 4 + p][w + kKernel    ][q];
              tmp13 +=
                weight_buf1[p][q] * input_buf1[h + 5 + p][w + kKernel    ][q];
              tmp14 +=
                weight_buf1[p][q] * input_buf1[h + 6 + p][w + kKernel    ][q];
              tmp15 +=
                weight_buf1[p][q] * input_buf1[h + 7 + p][w + kKernel    ][q];

            }
          }
          output_buf1[h    ][w    ] += tmp0; //store reduction result
          output_buf1[h + 1][w    ] += tmp1;
          output_buf1[h + 2][w    ] += tmp2;
          output_buf1[h + 3][w    ] += tmp3;
          output_buf1[h + 4][w    ] += tmp4;
          output_buf1[h + 5][w    ] += tmp5;
          output_buf1[h + 6][w    ] += tmp6;
          output_buf1[h + 7][w    ] += tmp7;
          output_buf1[h    ][w + 1] += tmp8;
          output_buf1[h + 1][w + 1] += tmp9;
          output_buf1[h + 2][w + 1] += tmp10;
          output_buf1[h + 3][w + 1] += tmp11;
          output_buf1[h + 4][w + 1] += tmp12;
          output_buf1[h + 5][w + 1] += tmp13;
          output_buf1[h + 6][w + 1] += tmp14;
          output_buf1[h + 7][w + 1] += tmp15;
        }
      }
    }
    //copy output here
    store1:
    for (int h = 0; h < kOutImSize; h++) {
      for (int w = 0; w < kOutImSize; w++) {
        output(i + 1, h, w) =  max(0, max(
                                      max(output_buf1[h * 2][w * 2    ], output_buf1[h * 2 + 1][w * 2    ]),
                                      max(output_buf1[h * 2][w * 2 + 1], output_buf1[h * 2 + 1][w * 2 + 1])));
      }
    }
  }
}
