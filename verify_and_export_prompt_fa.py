import torch
import torch_npu
import torch_npu.onnx
import math
import argparse
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper


# 注册自定义 ONNX symbolic，使导出时能识别该算子
def _pfa_symbolic(g, query, key, value, padding_mask, atten_mask, pse_shift,
                  actual_seq_lengths, deq_scale1, quant_scale1,
                  deq_scale2, quant_scale2, quant_offset2,
                  num_heads, scale_value, pre_tokens, next_tokens,
                  input_layout, num_key_value_heads, actual_seq_lengths_kv,
                  sparse_mode):
    # 从 graph 常量节点中解析出原始 Python 值
    num_heads_val = symbolic_helper._parse_arg(num_heads, "i")
    scale_value_val = symbolic_helper._parse_arg(scale_value, "f")
    pre_tokens_val = symbolic_helper._parse_arg(pre_tokens, "i")
    next_tokens_val = symbolic_helper._parse_arg(next_tokens, "i")
    input_layout_val = symbolic_helper._parse_arg(input_layout, "s")
    num_key_value_heads_val = symbolic_helper._parse_arg(num_key_value_heads, "i")

    # 过滤可选 tensor 输入: 只传非 None 的 tensor，避免 ATC 解析失败
    tensor_inputs = [query, key, value]
    for optional_input in [pse_shift, atten_mask, actual_seq_lengths]:
        if optional_input is not None and not symbolic_helper._is_none(optional_input):
            tensor_inputs.append(optional_input)

    return g.op("npu::NPUPromptFlashAttention",
                *tensor_inputs,
                num_heads_i=num_heads_val,
                scale_value_f=scale_value_val,
                pre_tokens_i=pre_tokens_val,
                next_tokens_i=next_tokens_val,
                input_layout_s=input_layout_val,
                num_key_value_heads_i=num_key_value_heads_val)


register_custom_op_symbolic("npu::npu_prompt_flash_attention", _pfa_symbolic, 1)


class Model(torch.nn.Module):
    def __init__(self, num_heads=8, head_dim=128, pre_tokens=65535, next_tokens=65535):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.scale_value = 1.0 / math.sqrt(float(head_dim))
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens

    def forward(self, q, k, v):
        # 直接调用底层 op，绕过 wrapper 签名不一致的问题
        res = torch.ops.npu.npu_prompt_flash_attention(
            q, k, v,
            num_heads=self.num_heads,
            scale_value=self.scale_value,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            input_layout="BNSD",
        )
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test npu_prompt_flash_attention")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX format")
    args = parser.parse_args()

    num_heads = 8
    head_dim = 128
    seq_len_q = 164
    seq_len_kv = 1024

    q = torch.randn(1, num_heads, seq_len_q, head_dim, dtype=torch.float16).npu()
    k = torch.randn(1, num_heads, seq_len_kv, head_dim, dtype=torch.float16).npu()
    v = torch.randn(1, num_heads, seq_len_kv, head_dim, dtype=torch.float16).npu()

    model = Model(num_heads=num_heads, head_dim=head_dim).to("npu")
    model.eval()

    out = model(q, k, v)
    print(f"outshape: {out.shape}, outdtype: {out.dtype}")

    if args.export_onnx:
        inputs = (q, k, v)
        with torch.no_grad():
            torch.onnx.export(model, inputs, "prompt_flash_attention_nodyn.onnx", dynamo=False)
        print("ONNX model exported to prompt_flash_attention_nodyn.onnx")
