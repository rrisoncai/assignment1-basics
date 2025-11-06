# Input configuration
batch_size = 1
seq_len = 1024 * 4
d_model = 1600
num_heads = 25
d_ff = 6400
num_layers = 48
vocab_size = 50257

print(f"| Parameter   | Value  |\n|-------------|--------|\n| batch_size  | {batch_size} |\n| seq_len     | {seq_len} |\n| d_model     | {d_model} |\n| d_ff        | {d_ff} |\n| num_layers  | {num_layers} |\n| vocab_size  | {vocab_size} |")

# Weights
w_token_embedding = vocab_size * d_model
w_position_embedding = seq_len * d_model

w_QKVO = num_layers * 4 * d_model * d_model
w_FFN = num_layers * 2 * d_model * d_ff
w_rmsnorm = num_layers * 5 * d_model

total_weights = w_QKVO + w_FFN + w_rmsnorm + w_token_embedding + w_position_embedding

print("\n| Component          | Parameters (Billion) | Percent of Total |")
print("|--------------------|----------------------|------------------|")
print(f"| Token Embedding    | {w_token_embedding / 1e9:.2f}               | {w_token_embedding / total_weights * 100:.2f}%           |")
print(f"| Position Embedding | {w_position_embedding / 1e9:.2f}               | {w_position_embedding / total_weights * 100:.2f}%           |")
print(f"| QKVO Projection    | {w_QKVO / 1e9:.2f}               | {w_QKVO / total_weights * 100:.2f}%           |")
print(f"| Feed-Forward       | {w_FFN / 1e9:.2f}               | {w_FFN / total_weights * 100:.2f}%           |")
print(f"| RMSNorm            | {w_rmsnorm / 1e9:.2f}               | {w_rmsnorm / total_weights * 100:.2f}%           |")
print(f"| **Total**          | {total_weights / 1e9:.2f}               | 100.00%          |")

# forward tops
f_mhsa = num_layers * 8 * batch_size * seq_len * d_model * d_model + 4 * batch_size * seq_len * seq_len * d_model
f_softmax = num_layers * 4 * batch_size * num_heads * seq_len * seq_len
f_ffn = num_layers * 6 * batch_size * seq_len * d_model * d_ff
f_linear = 2 * batch_size * seq_len * d_model * vocab_size

flops = f_mhsa + f_softmax + f_ffn + f_linear

print("\n| Component    | FLOPs (Tera) | Percent of Total |")
print("|--------------|--------------|------------------|")
print(f"| Self-Attn    | {f_mhsa / 1e12:.2f}       | {f_mhsa / flops * 100:.2f}%           |")
print(f"| Feed-Forward | {f_ffn / 1e12:.2f}       | {f_ffn / flops * 100:.2f}%           |")
print(f"| SoftMax      | {f_softmax / 1e12:.2f}       | {f_softmax / flops * 100:.2f}%           |")
print(f"| Linear       | {f_linear / 1e12:.2f}       | {f_linear / flops * 100:.2f}%           |")
print(f"| **Total**    | {flops / 1e12:.2f}       | 100.00%          |")