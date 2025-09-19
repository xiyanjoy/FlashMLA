import argparse
import math
from typing import Optional
import random

import torch
import triton

from flash_mla import flash_mla_with_kvcache, get_mla_metadata


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: tuple):
    # -1 means full extent
    return (group_shape[0] if group_shape[0] > 0 else x.shape[-2],
            group_shape[1] if group_shape[1] > 0 else x.shape[-1])


# Useful when treating N-dimensional group scaling as extended numpy-style
# broadcasting in numpy simply stretches dimensions with an extent of 1 to match
# the target shape by repeating the data along that dimension (broadcasting)
# , we extend these semantics to say if the extent of a dimension in the
# source shape is not 1 and does not match the target shape we repeat each
# element along that dimension src_shape[dim] // target_shape[dim] times
# example if we have:
#       a = [[1, 2], and target_shape = (2, 4)
#            [3, 4]]
# then we would expand a to:
#       a = [[1, 1, 2, 2],
#            [3, 3, 4, 4]]
# NOTE this function does not explicitly broadcast dimensions
# with an extent of 1, since this can be done implicitly by pytorch
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        if t.shape[i] != s and t.shape[i] != 1:
            assert s % t.shape[i] == 0
            t = t.unsqueeze(i + 1)\
                .expand(*t.shape[:i+1], s // t.shape[i], *t.shape[i+1:])\
                .flatten(i, i + 1)
    return t


# Quantize assuming once scale per group of elements with shape group_shape,
# example group shapes:
#  * (-1, -1)   for per-tensor quantization
#  * (1, -1)    for per-row quantization
#  * (-1, 1)    for per-column quantization
#  * (128, 128) for 128x128 deepseek style block quantization
#  * (1, 128)   for deepseek style activation quantization
#               (i.e. per-token-per-group)
def scaled_quantize(
    x: torch.Tensor,
    group_shape: tuple,
    quant_dtype: torch.dtype,
    scale: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_shape = _normalize_quant_group_shape(x, group_shape)
    assert quant_dtype.is_floating_point, \
        "currently `scaled_quantize` only supports floating point dtypes " \
        "but could be extended to support other dtypes"

    finfo = torch.finfo(quant_dtype)

    # Reshape (M, N) into (BLK_M, BLOCK_SIZE_M, BLK_N, BLOCK_SIZE_N)
    assert x.ndim == 2
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    x_blkd = x.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # Compute scales
    if scale is None:
        min_val, max_val = x_blkd_permd.aminmax(dim=-1)
        amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
        scale = finfo.max / amax
    else:
        scale = scale.reciprocal()
    assert scale.shape[0] == blk_m and scale.shape[1] == blk_n

    # Apply scale and convert form:
    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    x_scl_sat = (x_blkd_permd * scale.unsqueeze(-1))\
        .clamp(min=finfo.min, max=finfo.max)\
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])\
        .permute(0, 2, 1, 3)\
        .reshape(x.shape)

    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()


# inverses `scaled_quantize`
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: Optional[tuple] = None,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_shape is not None:
        group_shape = _normalize_quant_group_shape(x_q, group_shape)

    if x_s.ndim == 0:  # scalar
        x_s = x_s.unsqueeze(-1).unsqueeze(-1)  # convert to (1, 1) tensor
    if x_s.ndim == 1:
        if group_shape is None:
            raise AssertionError(
                "if x_s is 1D tensor, group_shape must be provided otherwise "
                "its ambiguous which dimension to broadcast x_s to")
        # unsqueeze the scales for the dimension where we want to broadcast
        # across the full extent
        if group_shape[0] == x_q.shape[-2]:
            x_s = x_s.unsqueeze(-2)
        elif group_shape[1] == x_q.shape[-1]:
            x_s = x_s.unsqueeze(-1)
        else:
            raise AssertionError(
                "if x_s is a vector we should be broadcasting it to the full "
                "extent of one of the dimensions")

    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * x_s).to(out_dtype)


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool=False) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}", flush=True)
    if use_fp8:
        assert cos_diff < 3e-2, f"cos_diff: {cos_diff} is too large"
    else:
        assert cos_diff < 1e-5, f"cos_diff: {cos_diff} is too large"

    return cos_diff

@torch.inference_mode()
def test_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen, torch_dtype):
    print(
        f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}, {torch_dtype=}"
    )
    use_fp8 = torch_dtype == torch.float8_e4m3fn
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q_float32 = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    blocked_k_float32 = torch.randn(block_table.numel(), block_size, h_kv, d)

    descale_k = None
    if use_fp8:
        fp8_dtype = torch.float8_e4m3fn
        _, descale_k = scaled_quantize(blocked_k_float32.reshape((-1, d)), (-1, 64), quant_dtype=fp8_dtype)
        descale_k = descale_k.reshape((d//64))

    for i in range(b):
        blocked_k_float32.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            float("nan")
        )
    blocked_v_float32 = blocked_k_float32[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv
    )

    def prepare_fp8_input():
        q_fp8, blocked_k_fp8, blocked_v_fp8, descale_q = None, None, None, None
        if use_fp8:
            nonlocal q_float32, blocked_k_float32, blocked_v_float32, descale_k
            fp8_dtype = torch.float8_e4m3fn

            q_fp8, descale_q = scaled_quantize(q_float32.reshape((-1, d)), (1, 64), quant_dtype=fp8_dtype)
            q_fp8 = q_fp8.reshape((b, s_q, h_q, d))
            descale_q = descale_q.reshape((b, s_q, h_q, d//64))

            blocked_k_fp8, _ = scaled_quantize(blocked_k_float32.reshape((-1, d)), (-1, 64), quant_dtype=fp8_dtype, scale=descale_k.unsqueeze(0))
            blocked_k_fp8 = blocked_k_fp8.reshape((block_table.numel(), block_size, h_kv, d))
            blocked_v_fp8 = blocked_k_fp8[..., :dv]

        return q_fp8, blocked_k_fp8, blocked_v_fp8, descale_q, descale_k

    q_fp8, blocked_k_fp8, blocked_v_fp8, descale_q, descale_k = prepare_fp8_input()
    if use_fp8:
        q = q_fp8
        blocked_k = blocked_k_fp8
        blocked_v = blocked_v_fp8
    else:
        q = q_float32.to(torch_dtype)
        blocked_k = blocked_k_float32.to(torch_dtype)
        blocked_v = blocked_v_float32.to(torch_dtype)

    def flash_mla():
        return flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )

    def ref_mla():
        q_ = q_float32
        blocked_k_ = blocked_k_float32
        blocked_v_ = blocked_v_float32
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q_[i].transpose(0, 1),
                blocked_k_.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v_.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = flash_mla()
    out_torch, lse_torch = ref_mla()
    cal_diff(out_flash.to(out_torch.dtype), out_torch, "out", use_fp8)
    cal_diff(lse_flash.to(lse_torch.dtype), lse_torch, "lse")

    t = triton.testing.do_bench(flash_mla)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
        torch.finfo(q.dtype).bits // 8
    )
    print(
        f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s",
        flush=True,
    )


def main(torch_dtype):
    device = torch.device("cuda:0")
    init_dtype = torch.float
    torch.set_default_dtype(init_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    h_kv = 1
    d, dv = 576, 512
    causal = True

    for b in [128]:
        for s in [4096, 8192, 16384]:
            for h_q in [16, 32, 64, 128]:  # TP = 8, 4, 2, 1
                for s_q in [1, 2]:  # MTP = 1, 2
                    for varlen in [False, True]:
                        test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen, torch_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "float8_e4m3fn"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16 or float8_e4m3fn)",
    )

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "float8_e4m3fn":
        torch_dtype = torch.float8_e4m3fn

    main(torch_dtype)
