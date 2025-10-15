from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
ch = "鸵"
ids = tokenizer.encode(ch, add_special_tokens=False)
print(ch, "→", ids)

# -*- coding: utf-8 -*-

utf8_bytes = ch.encode("utf-8")

print("字符:", ch)
print("UTF-8 字节:", utf8_bytes)
print("逐字节十六进制:", [hex(b) for b in utf8_bytes])

import tiktoken

# 选择 GPT-4 / GPT-3.5 系列所用的 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# 查看 token 113 对应的字节串和可打印形式
print("token id:", 113)
print("decoded bytes:", enc.decode_single_token_bytes(113))
print("decoded text:", enc.decode([113]))
