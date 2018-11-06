import torch as tc
import numpy as np
import mxnet as mx
mx_args = ['pre_conv_weight', 'pre_bn_gamma', 'pre_bn_beta', 'pre_res1_conv_weight', 'pre_res1_bn_gamma', 'pre_res1_bn_beta', 'pre_res2_conv_weight', 'pre_res2_bn_gamma', 'pre_res2_bn_beta', 'pre_res_skip_conv_weight', 'pre_res_skip_bn_gamma', 'pre_res_skip_bn_beta', 'up_0_1_5_0_res1_conv_weight', 'up_0_1_5_0_res1_bn_gamma', 'up_0_1_5_0_res1_bn_beta', 'up_0_1_5_0_res2_conv_weight', 'up_0_1_5_0_res2_bn_gamma', 'up_0_1_5_0_res2_bn_beta', 'up_0_1_5_1_res1_conv_weight', 'up_0_1_5_1_res1_bn_gamma', 'up_0_1_5_1_res1_bn_beta', 'up_0_1_5_1_res2_conv_weight', 'up_0_1_5_1_res2_bn_gamma', 'up_0_1_5_1_res2_bn_beta','low_0_1_5_0_res1_conv_weight', 'low_0_1_5_0_res1_bn_gamma', 'low_0_1_5_0_res1_bn_beta', 'low_0_1_5_0_res2_conv_weight', 'low_0_1_5_0_res2_bn_gamma', 'low_0_1_5_0_res2_bn_beta', 'low_0_1_5_0_res_skip_conv_weight', 'low_0_1_5_0_res_skip_bn_gamma', 'low_0_1_5_0_res_skip_bn_beta', 'low_0_1_5_1_res1_conv_weight', 'low_0_1_5_1_res1_bn_gamma', 'low_0_1_5_1_res1_bn_beta', 'low_0_1_5_1_res2_conv_weight', 'low_0_1_5_1_res2_bn_gamma', 'low_0_1_5_1_res2_bn_beta', 'up_0_1_4_0_res1_conv_weight', 'up_0_1_4_0_res1_bn_gamma', 'up_0_1_4_0_res1_bn_beta', 'up_0_1_4_0_res2_conv_weight', 'up_0_1_4_0_res2_bn_gamma', 'up_0_1_4_0_res2_bn_beta', 'up_0_1_4_1_res1_conv_weight', 'up_0_1_4_1_res1_bn_gamma', 'up_0_1_4_1_res1_bn_beta', 'up_0_1_4_1_res2_conv_weight', 'up_0_1_4_1_res2_bn_gamma', 'up_0_1_4_1_res2_bn_beta', 'low_0_1_4_0_res1_conv_weight', 'low_0_1_4_0_res1_bn_gamma', 'low_0_1_4_0_res1_bn_beta', 'low_0_1_4_0_res2_conv_weight', 'low_0_1_4_0_res2_bn_gamma', 'low_0_1_4_0_res2_bn_beta', 'low_0_1_4_0_res_skip_conv_weight', 'low_0_1_4_0_res_skip_bn_gamma', 'low_0_1_4_0_res_skip_bn_beta', 'low_0_1_4_1_res1_conv_weight', 'low_0_1_4_1_res1_bn_gamma', 'low_0_1_4_1_res1_bn_beta', 'low_0_1_4_1_res2_conv_weight', 'low_0_1_4_1_res2_bn_gamma', 'low_0_1_4_1_res2_bn_beta', 'up_0_1_3_0_res1_conv_weight', 'up_0_1_3_0_res1_bn_gamma', 'up_0_1_3_0_res1_bn_beta', 'up_0_1_3_0_res2_conv_weight', 'up_0_1_3_0_res2_bn_gamma', 'up_0_1_3_0_res2_bn_beta', 'up_0_1_3_1_res1_conv_weight', 'up_0_1_3_1_res1_bn_gamma', 'up_0_1_3_1_res1_bn_beta', 'up_0_1_3_1_res2_conv_weight', 'up_0_1_3_1_res2_bn_gamma', 'up_0_1_3_1_res2_bn_beta', 'low_0_1_3_0_res1_conv_weight', 'low_0_1_3_0_res1_bn_gamma', 'low_0_1_3_0_res1_bn_beta', 'low_0_1_3_0_res2_conv_weight', 'low_0_1_3_0_res2_bn_gamma', 'low_0_1_3_0_res2_bn_beta', 'low_0_1_3_0_res_skip_conv_weight', 'low_0_1_3_0_res_skip_bn_gamma', 'low_0_1_3_0_res_skip_bn_beta', 'low_0_1_3_1_res1_conv_weight', 'low_0_1_3_1_res1_bn_gamma', 'low_0_1_3_1_res1_bn_beta', 'low_0_1_3_1_res2_conv_weight', 'low_0_1_3_1_res2_bn_gamma', 'low_0_1_3_1_res2_bn_beta', 'up_0_1_2_0_res1_conv_weight', 'up_0_1_2_0_res1_bn_gamma', 'up_0_1_2_0_res1_bn_beta', 'up_0_1_2_0_res2_conv_weight', 'up_0_1_2_0_res2_bn_gamma', 'up_0_1_2_0_res2_bn_beta', 'up_0_1_2_1_res1_conv_weight', 'up_0_1_2_1_res1_bn_gamma', 'up_0_1_2_1_res1_bn_beta', 'up_0_1_2_1_res2_conv_weight', 'up_0_1_2_1_res2_bn_gamma', 'up_0_1_2_1_res2_bn_beta', 'low_0_1_2_0_res1_conv_weight', 'low_0_1_2_0_res1_bn_gamma', 'low_0_1_2_0_res1_bn_beta', 'low_0_1_2_0_res2_conv_weight', 'low_0_1_2_0_res2_bn_gamma', 'low_0_1_2_0_res2_bn_beta', 'low_0_1_2_0_res_skip_conv_weight', 'low_0_1_2_0_res_skip_bn_gamma', 'low_0_1_2_0_res_skip_bn_beta', 'low_0_1_2_1_res1_conv_weight', 'low_0_1_2_1_res1_bn_gamma', 'low_0_1_2_1_res1_bn_beta', 'low_0_1_2_1_res2_conv_weight', 'low_0_1_2_1_res2_bn_gamma', 'low_0_1_2_1_res2_bn_beta', 'up_0_1_1_0_res1_conv_weight', 'up_0_1_1_0_res1_bn_gamma', 'up_0_1_1_0_res1_bn_beta', 'up_0_1_1_0_res2_conv_weight', 'up_0_1_1_0_res2_bn_gamma', 'up_0_1_1_0_res2_bn_beta', 'up_0_1_1_1_res1_conv_weight', 'up_0_1_1_1_res1_bn_gamma', 'up_0_1_1_1_res1_bn_beta', 'up_0_1_1_1_res2_conv_weight', 'up_0_1_1_1_res2_bn_gamma', 'up_0_1_1_1_res2_bn_beta', 'low_0_1_1_0_res1_conv_weight', 'low_0_1_1_0_res1_bn_gamma', 'low_0_1_1_0_res1_bn_beta', 'low_0_1_1_0_res2_conv_weight', 'low_0_1_1_0_res2_bn_gamma', 'low_0_1_1_0_res2_bn_beta', 'low_0_1_1_0_res_skip_conv_weight', 'low_0_1_1_0_res_skip_bn_gamma', 'low_0_1_1_0_res_skip_bn_beta', 'low_0_1_1_1_res1_conv_weight', 'low_0_1_1_1_res1_bn_gamma', 'low_0_1_1_1_res1_bn_beta', 'low_0_1_1_1_res2_conv_weight', 'low_0_1_1_1_res2_bn_gamma', 'low_0_1_1_1_res2_bn_beta', 'low_0_2_1_0_res1_conv_weight', 'low_0_2_1_0_res1_bn_gamma', 'low_0_2_1_0_res1_bn_beta', 'low_0_2_1_0_res2_conv_weight', 'low_0_2_1_0_res2_bn_gamma', 'low_0_2_1_0_res2_bn_beta', 'low_0_2_1_1_res1_conv_weight', 'low_0_2_1_1_res1_bn_gamma', 'low_0_2_1_1_res1_bn_beta', 'low_0_2_1_1_res2_conv_weight', 'low_0_2_1_1_res2_bn_gamma', 'low_0_2_1_1_res2_bn_beta', 'low_0_2_1_2_res1_conv_weight', 'low_0_2_1_2_res1_bn_gamma', 'low_0_2_1_2_res1_bn_beta', 'low_0_2_1_2_res2_conv_weight', 'low_0_2_1_2_res2_bn_gamma', 'low_0_2_1_2_res2_bn_beta', 'low_0_2_1_3_res1_conv_weight', 'low_0_2_1_3_res1_bn_gamma', 'low_0_2_1_3_res1_bn_beta','low_0_2_1_3_res2_conv_weight', 'low_0_2_1_3_res2_bn_gamma', 'low_0_2_1_3_res2_bn_beta', 'low_0_3_1_0_res1_conv_weight', 'low_0_3_1_0_res1_bn_gamma', 'low_0_3_1_0_res1_bn_beta', 'low_0_3_1_0_res2_conv_weight', 'low_0_3_1_0_res2_bn_gamma', 'low_0_3_1_0_res2_bn_beta', 'low_0_3_1_1_res1_conv_weight', 'low_0_3_1_1_res1_bn_gamma', 'low_0_3_1_1_res1_bn_beta', 'low_0_3_1_1_res2_conv_weight', 'low_0_3_1_1_res2_bn_gamma', 'low_0_3_1_1_res2_bn_beta', 'low_0_3_1_1_res_skip_conv_weight', 'low_0_3_1_1_res_skip_bn_gamma', 'low_0_3_1_1_res_skip_bn_beta', 'low_0_3_2_0_res1_conv_weight', 'low_0_3_2_0_res1_bn_gamma', 'low_0_3_2_0_res1_bn_beta', 'low_0_3_2_0_res2_conv_weight', 'low_0_3_2_0_res2_bn_gamma', 'low_0_3_2_0_res2_bn_beta', 'low_0_3_2_1_res1_conv_weight', 'low_0_3_2_1_res1_bn_gamma', 'low_0_3_2_1_res1_bn_beta', 'low_0_3_2_1_res2_conv_weight', 'low_0_3_2_1_res2_bn_gamma', 'low_0_3_2_1_res2_bn_beta', 'low_0_3_3_0_res1_conv_weight', 'low_0_3_3_0_res1_bn_gamma', 'low_0_3_3_0_res1_bn_beta', 'low_0_3_3_0_res2_conv_weight', 'low_0_3_3_0_res2_bn_gamma', 'low_0_3_3_0_res2_bn_beta', 'low_0_3_3_1_res1_conv_weight', 'low_0_3_3_1_res1_bn_gamma', 'low_0_3_3_1_res1_bn_beta', 'low_0_3_3_1_res2_conv_weight', 'low_0_3_3_1_res2_bn_gamma', 'low_0_3_3_1_res2_bn_beta', 'low_0_3_4_0_res1_conv_weight', 'low_0_3_4_0_res1_bn_gamma', 'low_0_3_4_0_res1_bn_beta', 'low_0_3_4_0_res2_conv_weight', 'low_0_3_4_0_res2_bn_gamma', 'low_0_3_4_0_res2_bn_beta', 'low_0_3_4_1_res1_conv_weight', 'low_0_3_4_1_res1_bn_gamma', 'low_0_3_4_1_res1_bn_beta', 'low_0_3_4_1_res2_conv_weight', 'low_0_3_4_1_res2_bn_gamma', 'low_0_3_4_1_res2_bn_beta', 'low_0_3_4_1_res_skip_conv_weight', 'low_0_3_4_1_res_skip_bn_gamma', 'low_0_3_4_1_res_skip_bn_beta', 'low_0_3_5_0_res1_conv_weight', 'low_0_3_5_0_res1_bn_gamma', 'low_0_3_5_0_res1_bn_beta', 'low_0_3_5_0_res2_conv_weight', 'low_0_3_5_0_res2_bn_gamma', 'low_0_3_5_0_res2_bn_beta', 'low_0_3_5_1_res1_conv_weight', 'low_0_3_5_1_res1_bn_gamma', 'low_0_3_5_1_res1_bn_beta', 'low_0_3_5_1_res2_conv_weight', 'low_0_3_5_1_res2_bn_gamma', 'low_0_3_5_1_res2_bn_beta', 'cnv_0_conv_weight', 'cnv_0_bn_gamma', 'cnv_0_bn_beta', 'tl_0_skip_conv_weight', 'tl_0_skip_bn_gamma', 'tl_0_skip_bn_beta', 't_0_conv_weight', 't_0_bn_gamma', 't_0_bn_beta', 'l_0_conv_weight', 'l_0_bn_gamma', 'l_0_bn_beta', 'tl_0_p_conv_weight', 'tl_0_p_bn_gamma', 'tl_0_p_bn_beta', 'tl_0_out_conv_weight', 'tl_0_out_bn_gamma', 'tl_0_out_bn_beta', 'tl_0_heat1_conv_weight', 'tl_0_heat1_conv_bias', 'tl_0_heat_out_weight', 'tl_0_heat_out_bias', 'br_0_skip_conv_weight', 'br_0_skip_bn_gamma', 'br_0_skip_bn_beta', 'b_0_conv_weight', 'b_0_bn_gamma', 'b_0_bn_beta', 'r_0_conv_weight', 'r_0_bn_gamma', 'r_0_bn_beta', 'br_0_p_conv_weight', 'br_0_p_bn_gamma', 'br_0_p_bn_beta', 'br_0_out_conv_weight', 'br_0_out_bn_gamma', 'br_0_out_bn_beta', 'br_0_heat1_conv_weight', 'br_0_heat1_conv_bias', 'br_0_heat_out_weight', 'br_0_heat_out_bias', 'tl_0_tag1_conv_weight', 'tl_0_tag1_conv_bias', 'tl_0_tag_out_weight', 'tl_0_tag_out_bias', 'br_0_tag1_conv_weight', 'br_0_tag1_conv_bias', 'br_0_tag_out_weight', 'br_0_tag_out_bias', 'tl_0_regrs1_conv_weight', 'tl_0_regrs1_conv_bias', 'tl_0_regrs_out_weight', 'tl_0_regrs_out_bias', 'br_0_regrs1_conv_weight', 'br_0_regrs1_conv_bias', 'br_0_regrs_out_weight', 'br_0_regrs_out_bias', 'inter_0_conv_weight', 'inter_0_bn_gamma', 'inter_0_bn_beta', 'cnv_inter_0_conv_weight', 'cnv_inter_0_bn_gamma', 'cnv_inter_0_bn_beta', 'inter_0_res1_conv_weight', 'inter_0_res1_bn_gamma', 'inter_0_res1_bn_beta', 'inter_0_res2_conv_weight', 'inter_0_res2_bn_gamma', 'inter_0_res2_bn_beta', 'up_1_1_5_0_res1_conv_weight', 'up_1_1_5_0_res1_bn_gamma', 'up_1_1_5_0_res1_bn_beta', 'up_1_1_5_0_res2_conv_weight', 'up_1_1_5_0_res2_bn_gamma', 'up_1_1_5_0_res2_bn_beta', 'up_1_1_5_1_res1_conv_weight', 'up_1_1_5_1_res1_bn_gamma', 'up_1_1_5_1_res1_bn_beta', 'up_1_1_5_1_res2_conv_weight', 'up_1_1_5_1_res2_bn_gamma', 'up_1_1_5_1_res2_bn_beta', 'low_1_1_5_0_res1_conv_weight', 'low_1_1_5_0_res1_bn_gamma', 'low_1_1_5_0_res1_bn_beta', 'low_1_1_5_0_res2_conv_weight', 'low_1_1_5_0_res2_bn_gamma', 'low_1_1_5_0_res2_bn_beta', 'low_1_1_5_0_res_skip_conv_weight', 'low_1_1_5_0_res_skip_bn_gamma', 'low_1_1_5_0_res_skip_bn_beta', 'low_1_1_5_1_res1_conv_weight', 'low_1_1_5_1_res1_bn_gamma', 'low_1_1_5_1_res1_bn_beta', 'low_1_1_5_1_res2_conv_weight', 'low_1_1_5_1_res2_bn_gamma', 'low_1_1_5_1_res2_bn_beta', 'up_1_1_4_0_res1_conv_weight', 'up_1_1_4_0_res1_bn_gamma', 'up_1_1_4_0_res1_bn_beta', 'up_1_1_4_0_res2_conv_weight', 'up_1_1_4_0_res2_bn_gamma', 'up_1_1_4_0_res2_bn_beta', 'up_1_1_4_1_res1_conv_weight', 'up_1_1_4_1_res1_bn_gamma', 'up_1_1_4_1_res1_bn_beta', 'up_1_1_4_1_res2_conv_weight', 'up_1_1_4_1_res2_bn_gamma', 'up_1_1_4_1_res2_bn_beta', 'low_1_1_4_0_res1_conv_weight', 'low_1_1_4_0_res1_bn_gamma', 'low_1_1_4_0_res1_bn_beta', 'low_1_1_4_0_res2_conv_weight', 'low_1_1_4_0_res2_bn_gamma', 'low_1_1_4_0_res2_bn_beta', 'low_1_1_4_0_res_skip_conv_weight', 'low_1_1_4_0_res_skip_bn_gamma', 'low_1_1_4_0_res_skip_bn_beta', 'low_1_1_4_1_res1_conv_weight', 'low_1_1_4_1_res1_bn_gamma', 'low_1_1_4_1_res1_bn_beta', 'low_1_1_4_1_res2_conv_weight', 'low_1_1_4_1_res2_bn_gamma', 'low_1_1_4_1_res2_bn_beta', 'up_1_1_3_0_res1_conv_weight', 'up_1_1_3_0_res1_bn_gamma', 'up_1_1_3_0_res1_bn_beta', 'up_1_1_3_0_res2_conv_weight', 'up_1_1_3_0_res2_bn_gamma', 'up_1_1_3_0_res2_bn_beta', 'up_1_1_3_1_res1_conv_weight', 'up_1_1_3_1_res1_bn_gamma', 'up_1_1_3_1_res1_bn_beta', 'up_1_1_3_1_res2_conv_weight', 'up_1_1_3_1_res2_bn_gamma', 'up_1_1_3_1_res2_bn_beta', 'low_1_1_3_0_res1_conv_weight', 'low_1_1_3_0_res1_bn_gamma', 'low_1_1_3_0_res1_bn_beta', 'low_1_1_3_0_res2_conv_weight', 'low_1_1_3_0_res2_bn_gamma', 'low_1_1_3_0_res2_bn_beta', 'low_1_1_3_0_res_skip_conv_weight', 'low_1_1_3_0_res_skip_bn_gamma', 'low_1_1_3_0_res_skip_bn_beta', 'low_1_1_3_1_res1_conv_weight', 'low_1_1_3_1_res1_bn_gamma', 'low_1_1_3_1_res1_bn_beta', 'low_1_1_3_1_res2_conv_weight', 'low_1_1_3_1_res2_bn_gamma', 'low_1_1_3_1_res2_bn_beta', 'up_1_1_2_0_res1_conv_weight', 'up_1_1_2_0_res1_bn_gamma', 'up_1_1_2_0_res1_bn_beta', 'up_1_1_2_0_res2_conv_weight', 'up_1_1_2_0_res2_bn_gamma', 'up_1_1_2_0_res2_bn_beta', 'up_1_1_2_1_res1_conv_weight', 'up_1_1_2_1_res1_bn_gamma', 'up_1_1_2_1_res1_bn_beta', 'up_1_1_2_1_res2_conv_weight', 'up_1_1_2_1_res2_bn_gamma', 'up_1_1_2_1_res2_bn_beta', 'low_1_1_2_0_res1_conv_weight', 'low_1_1_2_0_res1_bn_gamma', 'low_1_1_2_0_res1_bn_beta', 'low_1_1_2_0_res2_conv_weight', 'low_1_1_2_0_res2_bn_gamma', 'low_1_1_2_0_res2_bn_beta', 'low_1_1_2_0_res_skip_conv_weight', 'low_1_1_2_0_res_skip_bn_gamma', 'low_1_1_2_0_res_skip_bn_beta', 'low_1_1_2_1_res1_conv_weight', 'low_1_1_2_1_res1_bn_gamma', 'low_1_1_2_1_res1_bn_beta', 'low_1_1_2_1_res2_conv_weight', 'low_1_1_2_1_res2_bn_gamma', 'low_1_1_2_1_res2_bn_beta', 'up_1_1_1_0_res1_conv_weight', 'up_1_1_1_0_res1_bn_gamma', 'up_1_1_1_0_res1_bn_beta', 'up_1_1_1_0_res2_conv_weight', 'up_1_1_1_0_res2_bn_gamma', 'up_1_1_1_0_res2_bn_beta', 'up_1_1_1_1_res1_conv_weight', 'up_1_1_1_1_res1_bn_gamma', 'up_1_1_1_1_res1_bn_beta', 'up_1_1_1_1_res2_conv_weight', 'up_1_1_1_1_res2_bn_gamma', 'up_1_1_1_1_res2_bn_beta', 'low_1_1_1_0_res1_conv_weight', 'low_1_1_1_0_res1_bn_gamma', 'low_1_1_1_0_res1_bn_beta', 'low_1_1_1_0_res2_conv_weight', 'low_1_1_1_0_res2_bn_gamma', 'low_1_1_1_0_res2_bn_beta', 'low_1_1_1_0_res_skip_conv_weight', 'low_1_1_1_0_res_skip_bn_gamma', 'low_1_1_1_0_res_skip_bn_beta', 'low_1_1_1_1_res1_conv_weight', 'low_1_1_1_1_res1_bn_gamma', 'low_1_1_1_1_res1_bn_beta', 'low_1_1_1_1_res2_conv_weight', 'low_1_1_1_1_res2_bn_gamma', 'low_1_1_1_1_res2_bn_beta', 'low_1_2_1_0_res1_conv_weight', 'low_1_2_1_0_res1_bn_gamma', 'low_1_2_1_0_res1_bn_beta', 'low_1_2_1_0_res2_conv_weight', 'low_1_2_1_0_res2_bn_gamma', 'low_1_2_1_0_res2_bn_beta', 'low_1_2_1_1_res1_conv_weight', 'low_1_2_1_1_res1_bn_gamma', 'low_1_2_1_1_res1_bn_beta', 'low_1_2_1_1_res2_conv_weight', 'low_1_2_1_1_res2_bn_gamma', 'low_1_2_1_1_res2_bn_beta', 'low_1_2_1_2_res1_conv_weight', 'low_1_2_1_2_res1_bn_gamma', 'low_1_2_1_2_res1_bn_beta', 'low_1_2_1_2_res2_conv_weight', 'low_1_2_1_2_res2_bn_gamma', 'low_1_2_1_2_res2_bn_beta', 'low_1_2_1_3_res1_conv_weight', 'low_1_2_1_3_res1_bn_gamma', 'low_1_2_1_3_res1_bn_beta', 'low_1_2_1_3_res2_conv_weight', 'low_1_2_1_3_res2_bn_gamma', 'low_1_2_1_3_res2_bn_beta', 'low_1_3_1_0_res1_conv_weight', 'low_1_3_1_0_res1_bn_gamma', 'low_1_3_1_0_res1_bn_beta', 'low_1_3_1_0_res2_conv_weight', 'low_1_3_1_0_res2_bn_gamma', 'low_1_3_1_0_res2_bn_beta', 'low_1_3_1_1_res1_conv_weight', 'low_1_3_1_1_res1_bn_gamma', 'low_1_3_1_1_res1_bn_beta', 'low_1_3_1_1_res2_conv_weight', 'low_1_3_1_1_res2_bn_gamma', 'low_1_3_1_1_res2_bn_beta', 'low_1_3_1_1_res_skip_conv_weight', 'low_1_3_1_1_res_skip_bn_gamma', 'low_1_3_1_1_res_skip_bn_beta', 'low_1_3_2_0_res1_conv_weight', 'low_1_3_2_0_res1_bn_gamma', 'low_1_3_2_0_res1_bn_beta', 'low_1_3_2_0_res2_conv_weight', 'low_1_3_2_0_res2_bn_gamma', 'low_1_3_2_0_res2_bn_beta', 'low_1_3_2_1_res1_conv_weight', 'low_1_3_2_1_res1_bn_gamma', 'low_1_3_2_1_res1_bn_beta', 'low_1_3_2_1_res2_conv_weight', 'low_1_3_2_1_res2_bn_gamma', 'low_1_3_2_1_res2_bn_beta', 'low_1_3_3_0_res1_conv_weight', 'low_1_3_3_0_res1_bn_gamma', 'low_1_3_3_0_res1_bn_beta', 'low_1_3_3_0_res2_conv_weight', 'low_1_3_3_0_res2_bn_gamma', 'low_1_3_3_0_res2_bn_beta', 'low_1_3_3_1_res1_conv_weight', 'low_1_3_3_1_res1_bn_gamma', 'low_1_3_3_1_res1_bn_beta', 'low_1_3_3_1_res2_conv_weight', 'low_1_3_3_1_res2_bn_gamma', 'low_1_3_3_1_res2_bn_beta', 'low_1_3_4_0_res1_conv_weight', 'low_1_3_4_0_res1_bn_gamma', 'low_1_3_4_0_res1_bn_beta', 'low_1_3_4_0_res2_conv_weight', 'low_1_3_4_0_res2_bn_gamma', 'low_1_3_4_0_res2_bn_beta', 'low_1_3_4_1_res1_conv_weight', 'low_1_3_4_1_res1_bn_gamma', 'low_1_3_4_1_res1_bn_beta', 'low_1_3_4_1_res2_conv_weight', 'low_1_3_4_1_res2_bn_gamma', 'low_1_3_4_1_res2_bn_beta', 'low_1_3_4_1_res_skip_conv_weight', 'low_1_3_4_1_res_skip_bn_gamma', 'low_1_3_4_1_res_skip_bn_beta', 'low_1_3_5_0_res1_conv_weight', 'low_1_3_5_0_res1_bn_gamma', 'low_1_3_5_0_res1_bn_beta', 'low_1_3_5_0_res2_conv_weight', 'low_1_3_5_0_res2_bn_gamma', 'low_1_3_5_0_res2_bn_beta', 'low_1_3_5_1_res1_conv_weight', 'low_1_3_5_1_res1_bn_gamma', 'low_1_3_5_1_res1_bn_beta', 'low_1_3_5_1_res2_conv_weight', 'low_1_3_5_1_res2_bn_gamma', 'low_1_3_5_1_res2_bn_beta', 'cnv_1_conv_weight', 'cnv_1_bn_gamma', 'cnv_1_bn_beta', 'tl_1_skip_conv_weight', 'tl_1_skip_bn_gamma', 'tl_1_skip_bn_beta', 't_1_conv_weight', 't_1_bn_gamma', 't_1_bn_beta', 'l_1_conv_weight', 'l_1_bn_gamma', 'l_1_bn_beta', 'tl_1_p_conv_weight', 'tl_1_p_bn_gamma', 'tl_1_p_bn_beta', 'tl_1_out_conv_weight', 'tl_1_out_bn_gamma', 'tl_1_out_bn_beta', 'tl_1_heat1_conv_weight', 'tl_1_heat1_conv_bias', 'tl_1_heat_out_weight', 'tl_1_heat_out_bias', 'br_1_skip_conv_weight', 'br_1_skip_bn_gamma', 'br_1_skip_bn_beta', 'b_1_conv_weight', 'b_1_bn_gamma', 'b_1_bn_beta', 'r_1_conv_weight', 'r_1_bn_gamma', 'r_1_bn_beta', 'br_1_p_conv_weight', 'br_1_p_bn_gamma', 'br_1_p_bn_beta', 'br_1_out_conv_weight', 'br_1_out_bn_gamma', 'br_1_out_bn_beta', 'br_1_heat1_conv_weight', 'br_1_heat1_conv_bias', 'br_1_heat_out_weight', 'br_1_heat_out_bias', 'tl_1_tag1_conv_weight', 'tl_1_tag1_conv_bias', 'tl_1_tag_out_weight', 'tl_1_tag_out_bias', 'br_1_tag1_conv_weight', 'br_1_tag1_conv_bias', 'br_1_tag_out_weight', 'br_1_tag_out_bias', 'tl_1_regrs1_conv_weight', 'tl_1_regrs1_conv_bias', 'tl_1_regrs_out_weight', 'tl_1_regrs_out_bias', 'br_1_regrs1_conv_weight', 'br_1_regrs1_conv_bias', 'br_1_regrs_out_weight', 'br_1_regrs_out_bias']
a = tc.load('/mnt/data-1/data/jiajie.tang/CornerNet_500000.pkl')
#mx_args = [_ for _ in mx_args if not 'moving' in _]
new_mx_dict = {}
mx_idx = 0
curr_mx_name = ''
mx_ref = mx.nd.load('cornernet-0000.params')
i = 0
val_map = {}
flags = {i:1 for i in mx_ref.keys()}
#for xx in mx_ref.keys():
#    print(xx)
for i, (k,v) in enumerate(a.items()):
    if 'inters.' in k :

        new_name = '_'.join(k.split('.')[1:])
        new_name = new_name.replace('running','moving').replace('conv1','res1_conv').replace('bn1','res1_bn').replace('conv2','res2_conv').replace('bn2','res2_bn').replace('bn_weight','bn_gamma').replace('bn_bias','bn_beta')
        if 'moving' in new_name:
            new_name = 'aux:' + new_name
        else:
            new_name = 'arg:' + new_name
        new_mx_dict[new_name ] = np.array(v)
        print(k,new_name)
        print(np.array(v).shape, mx_ref[new_name].shape)
        if (np.array(v).shape!= mx_ref[new_name].shape):
            import pdb
            pdb.set_trace()
            c = 1
        flags[new_name] = 0
        continue
        
    if 'inters_.' in k or 'cnvs_.' in k:

        new_name = '_'.join(k.split('.')[1:])
        new_name = new_name.replace('running','moving').replace('1_weight','1_gamma').replace('1_bias','1_beta').replace('inters_','inter').replace('cnvs_','cnvs')
        if 'moving' in new_name:
            new_name = 'aux:' + new_name
        else:
            new_name = 'arg:' + new_name
        new_mx_dict[new_name ] = np.array(v)
        print(k,new_name)
        print(np.array(v).shape, mx_ref[new_name].shape)
        if (np.array(v).shape!= mx_ref[new_name].shape):
            import pdb
            pdb.set_trace()
            c = 1
        flags[new_name] = 0
        continue
        
    if '_tags' in k or '_heats' in k or '_regrs' in k:
        new_name = '_'.join(k.split('.')[1:])
        new_name = new_name.replace('running','moving').replace('bn_weight','bn_gamma').replace('bn_bias','bn_beta')
        if 'moving' in new_name:
            new_name = 'aux:' + new_name
        else:
            new_name = 'arg:' + new_name
        new_mx_dict[new_name ] = np.array(v)
        print(k,new_name)
        print(np.array(v).shape, mx_ref[new_name].shape)
        if (np.array(v).shape!= mx_ref[new_name].shape):
            import pdb
            pdb.set_trace()
            c = 1
        flags[new_name] = 0
        continue
        
    if 'module.tl_cnvs' in k or 'module.br_cnvs' in k:
        new_name = '_'.join(k.split('.')[1:])
        new_name = new_name.replace('_cnvs','').replace('running','moving').replace('bn_weight','bn_gamma').replace('bn_bias','bn_beta')
        new_name = new_name.replace('bn1_weight','bn1_gamma').replace('bn1_bias','bn1_beta')
        if 'moving' in new_name:
            new_name = 'aux:' + new_name
        else:
            new_name = 'arg:' + new_name
        new_mx_dict[new_name ] = np.array(v)
        print(k,new_name)
        print(np.array(v).shape, mx_ref[new_name].shape)
        if (np.array(v).shape!= mx_ref[new_name].shape):
            import pdb
            pdb.set_trace()
            c = 1
        flags[new_name] = 0
        continue


    if 'module.cnvs' in k:
        new_name = '_'.join(k.split('.')[1:])
        new_name = new_name.replace('cnvs','cnv').replace('running','moving').replace('bn_weight','bn_gamma').replace('bn_bias','bn_beta')
        if 'moving' in new_name:
            new_name = 'aux:' + new_name
        else:
            new_name = 'arg:' + new_name
        new_mx_dict[new_name ] = np.array(v)
        print(k,new_name)
        print(np.array(v).shape, mx_ref[new_name].shape)
        if (np.array(v).shape!= mx_ref[new_name].shape):
            import pdb
            pdb.set_trace()
            c = 1
        flags[new_name] = 0
        continue
    if 'kps.1' in k:
        new_name = val_map[k.replace('kps.1','kps.0')]
        new_name = new_name.split('_')
        new_name[1] = str(1)
        new_name = '_'.join(new_name)
        new_mx_dict[new_name] = np.array(v)
        print(k,new_name)
        print(np.array(v).shape, mx_ref[new_name].shape)
        if (np.array(v).shape!= mx_ref[new_name].shape):
            import pdb
            pdb.set_trace()
            c = 1
        flags[new_name] = 0
        continue

    if 'running' in k:
        new_mx_dict['aux:'+curr_mx_name + '_'+k.split('.')[-1].replace('running','moving')] = np.array(v)
        val_map[k] = 'aux:'+curr_mx_name + '_'+k.split('.')[-1].replace('running','moving')
        print(np.array(v).shape, mx_ref['aux:'+curr_mx_name + '_'+k.split('.')[-1].replace('running','moving')].shape)
        print(k, 'aux:'+curr_mx_name +'_'+ k.split('.')[-1].replace('running','moving'))
        if (np.array(v).shape != mx_ref['aux:'+curr_mx_name + '_'+k.split('.')[-1].replace('running','moving')].shape):
            assert 0,'error!'
        flags['aux:'+curr_mx_name + '_'+k.split('.')[-1].replace('running','moving')] = 0
        continue
    else:
        new_mx_dict['arg:'+mx_args[mx_idx]] = np.array(v)
        val_map[k] = 'arg:'+mx_args[mx_idx]
        curr_mx_name = '_'.join(mx_args[mx_idx].split('_')[:-1])
        print(k, 'arg:'+mx_args[mx_idx])
        print(np.array(v).shape,mx_ref['arg:'+mx_args[mx_idx]].shape)
        if (np.array(v).shape!=mx_ref['arg:'+mx_args[mx_idx]].shape):
            assert 0,'error!'
        flags['arg:'+mx_args[mx_idx]] = 0
        mx_idx += 1
        
    
print([_ for _ in flags.keys() if flags[_] == 1])
new_mx_dict = {k:mx.nd.array(v) for k,v in new_mx_dict.items()}
mx.nd.save('pretrained_cornernet-0000.params',new_mx_dict)
    

