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
  float output_buf[kImSize][kImSize][4]
  __attribute__((xcl_array_partition(cyclic, 8, 1)))
  __attribute__((xcl_array_partition(cyclic, 2, 2)))
  __attribute__((xcl_array_partition(complete, 3)))
  ;

  float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
  __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
  __attribute__((xcl_array_partition(cyclic, 2, 2)))  // cyclic partition factor of 2 in dim 2 of input_buf
  __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
  ;

  float weight_buf[kKernel][kKernel][4] //buffer of weight
  __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
  __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
  __attribute__((xcl_array_partition(complete, 3)))
  ;

  for (int i = 0; i < kNum; i+=4) {
    //copy bias here
    load_bias:
    for (int h = 0; h < kImSize; h++) {
        for (int w = 0; w < kImSize; w++) {
          output_buf[h][w][0] = bias[i    ];
          output_buf[h][w][1] = bias[i + 1];
          output_buf[h][w][2] = bias[i + 2];
          output_buf[h][w][3] = bias[i + 3];
        }
    }
    for (int j = 0; j < kNum; j++) {
      //copy weight here
      load_weight:
      __attribute__((xcl_pipeline_loop))
      for (int p = 0; p < kKernel; p++) {
        for (int q = 0; q < kKernel; q++) {
          weight_buf[p][q][0] = weight(i    , j, p, q);
          weight_buf[p][q][1] = weight(i + 1, j, p, q);
          weight_buf[p][q][2] = weight(i + 2, j, p, q);
          weight_buf[p][q][3] = weight(i + 3, j, p, q);
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
          float tmp16 = 0;
          float tmp17 = 0;
          float tmp18 = 0;
          float tmp19 = 0;
          float tmp20 = 0;
          float tmp21 = 0;
          float tmp22 = 0;
          float tmp23 = 0;
          float tmp24 = 0;
          float tmp25 = 0;
          float tmp26 = 0;
          float tmp27 = 0;
          float tmp28 = 0;
          float tmp29 = 0;
          float tmp30 = 0;
          float tmp31 = 0;
          float tmp32 = 0;
          float tmp33 = 0;
          float tmp34 = 0;
          float tmp35 = 0;
          float tmp36 = 0;
          float tmp37 = 0;
          float tmp38 = 0;
          float tmp39 = 0;
          float tmp40 = 0;
          float tmp41 = 0;
          float tmp42 = 0;
          float tmp43 = 0;
          float tmp44 = 0;
          float tmp45 = 0;
          float tmp46 = 0;
          float tmp47 = 0;
          float tmp48 = 0;
          float tmp49 = 0;
          float tmp50 = 0;
          float tmp51 = 0;
          float tmp52 = 0;
          float tmp53 = 0;
          float tmp54 = 0;
          float tmp55 = 0;
          float tmp56 = 0;
          float tmp57 = 0;
          float tmp58 = 0;
          float tmp59 = 0;
          float tmp60 = 0;
          float tmp61 = 0;
          float tmp62 = 0;
          float tmp63 = 0;
          for (int p = 0; p < kKernel; p++) {  // unrolled loop
            for (int q = 0; q < kKernel; q++) {  //unrolled loop
              tmp0 += //will be synthesized into tree reduction
                weight_buf[p][q][0] * input_buf[h     + p][w + kKernel - 1][q];
              tmp1 +=
                weight_buf[p][q][0] * input_buf[h + 1 + p][w + kKernel - 1][q];
              tmp2 +=
                weight_buf[p][q][0] * input_buf[h + 2 + p][w + kKernel - 1][q];
              tmp3 +=
                weight_buf[p][q][0] * input_buf[h + 3 + p][w + kKernel - 1][q];
              tmp4 +=
                weight_buf[p][q][0] * input_buf[h + 4 + p][w + kKernel - 1][q];
              tmp5 +=
                weight_buf[p][q][0] * input_buf[h + 5 + p][w + kKernel - 1][q];
              tmp6 +=
                weight_buf[p][q][0] * input_buf[h + 6 + p][w + kKernel - 1][q];
              tmp7 +=
                weight_buf[p][q][0] * input_buf[h + 7 + p][w + kKernel - 1][q];
              tmp8 +=
                weight_buf[p][q][0] * input_buf[h     + p][w + kKernel    ][q];
              tmp9 +=
                weight_buf[p][q][0] * input_buf[h + 1 + p][w + kKernel    ][q];
              tmp10 +=
                weight_buf[p][q][0] * input_buf[h + 2 + p][w + kKernel    ][q];
              tmp11 +=
                weight_buf[p][q][0] * input_buf[h + 3 + p][w + kKernel    ][q];
              tmp12 +=
                weight_buf[p][q][0] * input_buf[h + 4 + p][w + kKernel    ][q];
              tmp13 +=
                weight_buf[p][q][0] * input_buf[h + 5 + p][w + kKernel    ][q];
              tmp14 +=
                weight_buf[p][q][0] * input_buf[h + 6 + p][w + kKernel    ][q];
              tmp15 +=
                weight_buf[p][q][0] * input_buf[h + 7 + p][w + kKernel    ][q];
              tmp16 +=
                weight_buf[p][q][1] * input_buf[h     + p][w + kKernel - 1][q];
              tmp17 +=
                weight_buf[p][q][1] * input_buf[h + 1 + p][w + kKernel - 1][q];
              tmp18 +=
                weight_buf[p][q][1] * input_buf[h + 2 + p][w + kKernel - 1][q];
              tmp19 +=
                weight_buf[p][q][1] * input_buf[h + 3 + p][w + kKernel - 1][q];
              tmp20 +=
                weight_buf[p][q][1] * input_buf[h + 4 + p][w + kKernel - 1][q];
              tmp21 +=
                weight_buf[p][q][1] * input_buf[h + 5 + p][w + kKernel - 1][q];
              tmp22 +=
                weight_buf[p][q][1] * input_buf[h + 6 + p][w + kKernel - 1][q];
              tmp23 +=
                weight_buf[p][q][1] * input_buf[h + 7 + p][w + kKernel - 1][q];
              tmp24 +=
                weight_buf[p][q][1] * input_buf[h     + p][w + kKernel    ][q];
              tmp25 +=
                weight_buf[p][q][1] * input_buf[h + 1 + p][w + kKernel    ][q];
              tmp26 +=
                weight_buf[p][q][1] * input_buf[h + 2 + p][w + kKernel    ][q];
              tmp27 +=
                weight_buf[p][q][1] * input_buf[h + 3 + p][w + kKernel    ][q];
              tmp28 +=
                weight_buf[p][q][1] * input_buf[h + 4 + p][w + kKernel    ][q];
              tmp29 +=
                weight_buf[p][q][1] * input_buf[h + 5 + p][w + kKernel    ][q];
              tmp30 +=
                weight_buf[p][q][1] * input_buf[h + 6 + p][w + kKernel    ][q];
              tmp31 +=
                weight_buf[p][q][1] * input_buf[h + 7 + p][w + kKernel    ][q];
              tmp32 +=
                weight_buf[p][q][2] * input_buf[h     + p][w + kKernel - 1][q];
              tmp33 +=
                weight_buf[p][q][2] * input_buf[h + 1 + p][w + kKernel - 1][q];
              tmp34 +=
                weight_buf[p][q][2] * input_buf[h + 2 + p][w + kKernel - 1][q];
              tmp35 +=
                weight_buf[p][q][2] * input_buf[h + 3 + p][w + kKernel - 1][q];
              tmp36 +=
                weight_buf[p][q][2] * input_buf[h + 4 + p][w + kKernel - 1][q];
              tmp37 +=
                weight_buf[p][q][2] * input_buf[h + 5 + p][w + kKernel - 1][q];
              tmp38 +=
                weight_buf[p][q][2] * input_buf[h + 6 + p][w + kKernel - 1][q];
              tmp39 +=
                weight_buf[p][q][2] * input_buf[h + 7 + p][w + kKernel - 1][q];
              tmp40 +=
                weight_buf[p][q][2] * input_buf[h     + p][w + kKernel    ][q];
              tmp41 +=
                weight_buf[p][q][2] * input_buf[h + 1 + p][w + kKernel    ][q];
              tmp42 +=
                weight_buf[p][q][2] * input_buf[h + 2 + p][w + kKernel    ][q];
              tmp43 +=
                weight_buf[p][q][2] * input_buf[h + 3 + p][w + kKernel    ][q];
              tmp44 +=
                weight_buf[p][q][2] * input_buf[h + 4 + p][w + kKernel    ][q];
              tmp45 +=
                weight_buf[p][q][2] * input_buf[h + 5 + p][w + kKernel    ][q];
              tmp46 +=
                weight_buf[p][q][2] * input_buf[h + 6 + p][w + kKernel    ][q];
              tmp47 +=
                weight_buf[p][q][2] * input_buf[h + 7 + p][w + kKernel    ][q];
              tmp48 +=
                weight_buf[p][q][3] * input_buf[h     + p][w + kKernel - 1][q];
              tmp49 +=
                weight_buf[p][q][3] * input_buf[h + 1 + p][w + kKernel - 1][q];
              tmp50 +=
                weight_buf[p][q][3] * input_buf[h + 2 + p][w + kKernel - 1][q];
              tmp51 +=
                weight_buf[p][q][3] * input_buf[h + 3 + p][w + kKernel - 1][q];
              tmp52 +=
                weight_buf[p][q][3] * input_buf[h + 4 + p][w + kKernel - 1][q];
              tmp53 +=
                weight_buf[p][q][3] * input_buf[h + 5 + p][w + kKernel - 1][q];
              tmp54 +=
                weight_buf[p][q][3] * input_buf[h + 6 + p][w + kKernel - 1][q];
              tmp55 +=
                weight_buf[p][q][3] * input_buf[h + 7 + p][w + kKernel - 1][q];
              tmp56 +=
                weight_buf[p][q][3] * input_buf[h     + p][w + kKernel    ][q];
              tmp57 +=
                weight_buf[p][q][3] * input_buf[h + 1 + p][w + kKernel    ][q];
              tmp58 +=
                weight_buf[p][q][3] * input_buf[h + 2 + p][w + kKernel    ][q];
              tmp59 +=
                weight_buf[p][q][3] * input_buf[h + 3 + p][w + kKernel    ][q];
              tmp60 +=
                weight_buf[p][q][3] * input_buf[h + 4 + p][w + kKernel    ][q];
              tmp61 +=
                weight_buf[p][q][3] * input_buf[h + 5 + p][w + kKernel    ][q];
              tmp62 +=
                weight_buf[p][q][3] * input_buf[h + 6 + p][w + kKernel    ][q];
              tmp63 +=
                weight_buf[p][q][3] * input_buf[h + 7 + p][w + kKernel    ][q];
            }
          }
          output_buf[h    ][w    ][0] += tmp0; //store reduction result
          output_buf[h + 1][w    ][0] += tmp1;
          output_buf[h + 2][w    ][0] += tmp2;
          output_buf[h + 3][w    ][0] += tmp3;
          output_buf[h + 4][w    ][0] += tmp4;
          output_buf[h + 5][w    ][0] += tmp5;
          output_buf[h + 6][w    ][0] += tmp6;
          output_buf[h + 7][w    ][0] += tmp7;
          output_buf[h    ][w + 1][0] += tmp8;
          output_buf[h + 1][w + 1][0] += tmp9;
          output_buf[h + 2][w + 1][0] += tmp10;
          output_buf[h + 3][w + 1][0] += tmp11;
          output_buf[h + 4][w + 1][0] += tmp12;
          output_buf[h + 5][w + 1][0] += tmp13;
          output_buf[h + 6][w + 1][0] += tmp14;
          output_buf[h + 7][w + 1][0] += tmp15;
          output_buf[h    ][w    ][1] += tmp16;
          output_buf[h + 1][w    ][1] += tmp17;
          output_buf[h + 2][w    ][1] += tmp18;
          output_buf[h + 3][w    ][1] += tmp19;
          output_buf[h + 4][w    ][1] += tmp20;
          output_buf[h + 5][w    ][1] += tmp21;
          output_buf[h + 6][w    ][1] += tmp22;
          output_buf[h + 7][w    ][1] += tmp23;
          output_buf[h    ][w + 1][1] += tmp24;
          output_buf[h + 1][w + 1][1] += tmp25;
          output_buf[h + 2][w + 1][1] += tmp26;
          output_buf[h + 3][w + 1][1] += tmp27;
          output_buf[h + 4][w + 1][1] += tmp28;
          output_buf[h + 5][w + 1][1] += tmp29;
          output_buf[h + 6][w + 1][1] += tmp30;
          output_buf[h + 7][w + 1][1] += tmp31;
          output_buf[h    ][w    ][2] += tmp32;
          output_buf[h + 1][w    ][2] += tmp33;
          output_buf[h + 2][w    ][2] += tmp34;
          output_buf[h + 3][w    ][2] += tmp35;
          output_buf[h + 4][w    ][2] += tmp36;
          output_buf[h + 5][w    ][2] += tmp37;
          output_buf[h + 6][w    ][2] += tmp38;
          output_buf[h + 7][w    ][2] += tmp39;
          output_buf[h    ][w + 1][2] += tmp40;
          output_buf[h + 1][w + 1][2] += tmp41;
          output_buf[h + 2][w + 1][2] += tmp42;
          output_buf[h + 3][w + 1][2] += tmp43;
          output_buf[h + 4][w + 1][2] += tmp44;
          output_buf[h + 5][w + 1][2] += tmp45;
          output_buf[h + 6][w + 1][2] += tmp46;
          output_buf[h + 7][w + 1][2] += tmp47;
          output_buf[h    ][w    ][3] += tmp48;
          output_buf[h + 1][w    ][3] += tmp49;
          output_buf[h + 2][w    ][3] += tmp50;
          output_buf[h + 3][w    ][3] += tmp51;
          output_buf[h + 4][w    ][3] += tmp52;
          output_buf[h + 5][w    ][3] += tmp53;
          output_buf[h + 6][w    ][3] += tmp54;
          output_buf[h + 7][w    ][3] += tmp55;
          output_buf[h    ][w + 1][3] += tmp56;
          output_buf[h + 1][w + 1][3] += tmp57;
          output_buf[h + 2][w + 1][3] += tmp58;
          output_buf[h + 3][w + 1][3] += tmp59;
          output_buf[h + 4][w + 1][3] += tmp60;
          output_buf[h + 5][w + 1][3] += tmp61;
          output_buf[h + 6][w + 1][3] += tmp62;
          output_buf[h + 7][w + 1][3] += tmp63;
        }
      }
    }
    //copy output here
    store:
    for (int h = 0; h < kOutImSize; h++) {
      for (int w = 0; w < kOutImSize; w++) {
        output(i    , h, w) =  max(0, max(
                                          max(output_buf[h * 2][w * 2    ][0], output_buf[h * 2 + 1][w * 2    ][0]),
                                          max(output_buf[h * 2][w * 2 + 1][0], output_buf[h * 2 + 1][w * 2 + 1][0])));
        output(i + 1, h, w) =  max(0, max(
                                          max(output_buf[h * 2][w * 2    ][1], output_buf[h * 2 + 1][w * 2    ][1]),
                                          max(output_buf[h * 2][w * 2 + 1][1], output_buf[h * 2 + 1][w * 2 + 1][1])));
        output(i + 2, h, w) =  max(0, max(
                                          max(output_buf[h * 2][w * 2    ][2], output_buf[h * 2 + 1][w * 2    ][2]),
                                          max(output_buf[h * 2][w * 2 + 1][2], output_buf[h * 2 + 1][w * 2 + 1][2])));
        output(i + 3, h, w) =  max(0, max(
                                          max(output_buf[h * 2][w * 2    ][3], output_buf[h * 2 + 1][w * 2    ][3]),
                                          max(output_buf[h * 2][w * 2 + 1][3], output_buf[h * 2 + 1][w * 2 + 1][3])));
      }
    }
  }
}
