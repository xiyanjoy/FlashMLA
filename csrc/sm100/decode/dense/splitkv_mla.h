#pragma once

#include "params.h"

namespace sm100 {

void run_flash_splitkv_mla_dense_kernel(DecodingParams &params, cudaStream_t stream);

}

