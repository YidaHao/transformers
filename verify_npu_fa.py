"""
最小验证脚本:检查 NPU Flash Attention 是否能在当前开发板上正常工作。

运行:
    python verify_npu_fa.py

验证内容:
    1. torch_npu / npu_fusion_attention 是否可用
    2. npu_flash_attn_func (定长 BSND) 前向是否能跑通，并与 eager 注意力对比数值
    3. npu_flash_attn_varlen_func (变长 TND，padding-free) 前向是否能跑通
"""

import math

import torch


def eager_causal_attention(q, k, v, scale, causal):
    """参考实现：标准 (B, H, S, D) 注意力，用于数值对比。"""
    # q,k,v: (B, S, H, D) -> (B, H, S, D)
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        s = q.shape[-2]
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)  # (B, H, S, D)
    return out.transpose(1, 2)  # (B, S, H, D)


def main():
    # ---------- 1. 环境检查 ----------
    try:
        import torch_npu  # noqa: F401
        from transformers.utils.import_utils import is_torch_npu_available
    except ImportError as e:
        raise SystemExit(f"[FAIL] 无法导入 torch_npu / transformers: {e}")

    if not is_torch_npu_available():
        raise SystemExit("[FAIL] is_torch_npu_available() 返回 False，当前环境没有可用的 NPU。")

    from transformers.integrations.npu_flash_attention import (
        npu_flash_attn_func,
        npu_flash_attn_varlen_func,
    )

    device = "npu:0"
    dtype = torch.float16
    print(f"[OK] NPU 可用，使用 device={device}, dtype={dtype}")
    print(f"[INFO] NPU device count: {torch.npu.device_count()}")

    # ---------- 2. 定长 npu_flash_attn_func ----------
    B, S, H, D = 2, 128, 8, 64
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)

    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    out = npu_flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
    assert out.shape == (B, S, H, D), f"输出形状错误: {out.shape}"
    assert torch.isfinite(out).all(), "输出包含 NaN/Inf"

    ref = eager_causal_attention(q, k, v, scale, causal=True).to(dtype)
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"[OK] npu_flash_attn_func 跑通，shape={tuple(out.shape)}，与 eager 最大误差={max_diff:.4e}")
    if max_diff > 2e-2:
        print(f"[WARN] 数值误差偏大 ({max_diff:.4e})，请确认 causal mask 对齐方式 (NPU_FA2_SPARSE_MODE)。")

    # ---------- 3. 变长 npu_flash_attn_varlen_func (padding-free) ----------
    # 两条序列拼成一个 batch：长度 100 + 28 = 128
    seqlens = [100, 28]
    total = sum(seqlens)
    cu_seqlens = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device=device, dtype=torch.int32)

    qv = torch.randn(total, H, D, device=device, dtype=dtype)
    kv = torch.randn(total, H, D, device=device, dtype=dtype)
    vv = torch.randn(total, H, D, device=device, dtype=dtype)

    out_var = npu_flash_attn_varlen_func(
        qv, kv, vv,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens),
        max_seqlen_k=max(seqlens),
        softmax_scale=scale,
        causal=True,
    )
    assert out_var.shape == (total, H, D), f"varlen 输出形状错误: {out_var.shape}"
    assert torch.isfinite(out_var).all(), "varlen 输出包含 NaN/Inf"
    print(f"[OK] npu_flash_attn_varlen_func 跑通，shape={tuple(out_var.shape)}")

    print("\n[SUCCESS] NPU Flash Attention 在当前开发板上工作正常 ✅")


if __name__ == "__main__":
    main()
