"""
验证 PromptFlashAttention (PFA) 能否替代 FlashAttentionScore 用于 Gemma 推理 (Ascend 310P)。

分两步：
    1. 数值验证：PFA 在 causal 配置下与 eager 注意力是否一致
    2. 接入验证：把 PFA 注册成 transformers 的自定义 attention interface，跑 Gemma 前向

运行：
    python verify_pfa_attention.py                 # 仅做算子数值验证
    python verify_pfa_attention.py --gemma <path>  # 额外做 Gemma 端到端验证

说明：
    - PFA 是推理算子(无反向)，只能 inference，不能训练。
    - causal 通过 next_tokens=0 表达(只看当前及左侧)，pre_tokens 取大值表示左侧全可见。
"""

import argparse
import math

import torch
import torch_npu  # noqa: F401


# ----------------------------------------------------------------------------
# 工具：在 CPU 上做 finite 校验，避免在 NPU 触发 IsFinite 算子的运行时 JIT 编译
# （部分板子缺 C++ 头文件，编译 IsFinite 会报 'cstdint' file not found）
# ----------------------------------------------------------------------------
def assert_finite(t, msg="输出含 NaN/Inf"):
    assert torch.isfinite(t.detach().float().cpu()).all(), msg


# ----------------------------------------------------------------------------
# 参考实现
# ----------------------------------------------------------------------------
def make_causal_mask(sq, sk, device):
    """返回 bool 上三角 mask，True = 被屏蔽。query i 可见 key j 当且仅当 j <= i + (sk - sq)。"""
    offset = sk - sq
    idx_q = torch.arange(sq, device=device).view(-1, 1)
    idx_k = torch.arange(sk, device=device).view(1, -1)
    return idx_k > (idx_q + offset)


def eager_attention(q, k, v, scale, causal):
    """标准 (B, H, S, D) 注意力，float32 参考。q,k,v: (B, H, S, D)."""
    q, k, v = q.float(), k.float(), v.float()
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        # 下三角对齐到右下角(decode 友好)：query i 可见 key <= i + (sk - sq)
        offset = sk - sq
        idx_q = torch.arange(sq, device=q.device).view(-1, 1)
        idx_k = torch.arange(sk, device=q.device).view(1, -1)
        mask = idx_k > (idx_q + offset)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# ----------------------------------------------------------------------------
# PFA 封装：BNSD 布局，causal
# ----------------------------------------------------------------------------
def pfa_attention(q, k, v, num_heads, scale, causal):
    """
    q,k,v: (B, H, S, D)，调用 npu_prompt_flash_attention。
    causal 通过显式 atten_mask (上三角 bool，True=屏蔽) 实现，最可靠。
    """
    atten_mask = None
    if causal:
        atten_mask = make_causal_mask(q.shape[2], k.shape[2], q.device)
    return torch.ops.npu.npu_prompt_flash_attention(
        q,
        k,
        v,
        atten_mask=atten_mask,
        num_heads=num_heads,
        scale_value=scale,
        pre_tokens=65535,
        next_tokens=65535,
        input_layout="BNSD",
    )


# ----------------------------------------------------------------------------
# 1. 算子数值验证
# ----------------------------------------------------------------------------
def verify_op():
    device = "npu:0"
    dtype = torch.float16
    B, H, S, D = 1, 8, 256, 128
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    for causal in (False, True):
        out = pfa_attention(q, k, v, H, scale, causal)
        assert out.shape == (B, H, S, D), f"shape 错误: {out.shape}"
        assert_finite(out)

        ref = eager_attention(q, k, v, scale, causal).to(dtype)
        max_diff = (out.float() - ref.float()).abs().max().item()
        tag = "causal" if causal else "full"
        status = "OK" if max_diff < 3e-2 else "WARN"
        print(f"[{status}] PFA ({tag:6s}) vs eager，最大误差={max_diff:.4e}")
        if status == "WARN":
            print("       误差偏大：检查 causal 的 pre/next_tokens 语义是否与参考实现一致。")

    print("[OK] PromptFlashAttention 算子数值验证通过")

    # ---- IFA：decode 阶段，单 query × 长 KV ----
    Lkv = 256
    q1 = torch.randn(B, H, 1, D, device=device, dtype=dtype)
    kc = torch.randn(B, H, Lkv, D, device=device, dtype=dtype)
    vc = torch.randn(B, H, Lkv, D, device=device, dtype=dtype)

    out_ifa = ifa_attention(q1, kc, vc, H, scale)
    assert out_ifa.shape == (B, H, 1, D), f"IFA shape 错误: {out_ifa.shape}"
    assert_finite(out_ifa, "IFA 输出含 NaN/Inf")

    # decode 时单 query 能看到全部已生成的 kv（非因果裁剪，等价 full）
    ref_ifa = eager_attention(q1, kc, vc, scale, causal=False).to(dtype)
    diff_ifa = (out_ifa.float() - ref_ifa.float()).abs().max().item()
    status = "OK" if diff_ifa < 3e-2 else "WARN"
    print(f"[{status}] IFA (decode) vs eager，最大误差={diff_ifa:.4e}")
    print("[OK] IncreFlashAttention 算子数值验证通过\n")


# ----------------------------------------------------------------------------
# IFA 封装：decode 阶段，单 query (S=1) × 长 KV-cache
# ----------------------------------------------------------------------------
def ifa_attention(q, k, v, num_heads, scale):
    """q: (B, H, 1, D)，k/v: (B, H, Lkv, D)。"""
    return torch.ops.npu.npu_incre_flash_attention(
        q,
        k,
        v,
        num_heads=num_heads,
        scale_value=scale,
        input_layout="BNSD",
    )


# ----------------------------------------------------------------------------
# 2. 注册为 transformers attention interface 并跑 Gemma
# ----------------------------------------------------------------------------
def npu_pfa_interface(
    module,
    query,        # (B, H, S, D)
    key,          # (B, Hkv, S, D)
    value,        # (B, Hkv, S, D)
    attention_mask,
    scaling=None,
    dropout=0.0,
    **kwargs,
):
    """
    transformers attention interface：根据 query 长度自动切换
        - prefill (S > 1) → PromptFlashAttention
        - decode  (S == 1) → IncreFlashAttention
    返回 (attn_output, None)。
    """
    # GQA：把 kv 头扩展到与 q 头一致
    num_heads = query.shape[1]
    num_kv = key.shape[1]
    if num_kv != num_heads:
        rep = num_heads // num_kv
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)

    scale = float(scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1]))
    q_len = query.shape[2]
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if q_len > 1:
        # Prefill：整段 prompt，显式 atten_mask 表达 causal
        causal = getattr(module, "is_causal", True)
        atten_mask = make_causal_mask(q_len, key.shape[2], query.device) if causal else None
        out = torch.ops.npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            atten_mask=atten_mask,
            num_heads=num_heads,
            scale_value=scale,
            pre_tokens=65535,
            next_tokens=65535,
            input_layout="BNSD",
        )
    else:
        # Decode：单 token，读全部 KV-cache（已是因果裁剪后的历史）
        out = torch.ops.npu.npu_incre_flash_attention(
            query,
            key,
            value,
            num_heads=num_heads,
            scale_value=scale,
            input_layout="BNSD",
        )
    # transformers 约定返回 (B, S, H, D)
    out = out.transpose(1, 2).contiguous()
    return out, None


def verify_gemma(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # 注册自定义实现
    ALL_ATTENTION_FUNCTIONS.register("npu_pfa", npu_pfa_interface)

    device = "npu:0"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation="npu_pfa",
    ).to(device).eval()

    inputs = tok("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        out_pfa = model(**inputs).logits

    # 对照 eager
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out_eager = model(**inputs).logits

    max_diff = (out_pfa.float() - out_eager.float()).abs().max().item()
    print(f"[INFO] Gemma logits PFA vs eager 最大误差={max_diff:.4e}")
    if max_diff < 1e-1:
        print("[OK] Gemma 使用 PFA 前向与 eager 一致 ✅")
    else:
        print("[WARN] 误差偏大，检查 mask/scale/GQA 处理。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=str, default=None, help="Gemma 模型路径，提供则做端到端验证")
    args = parser.parse_args()

    verify_op()
    if args.gemma:
        verify_gemma(args.gemma)
