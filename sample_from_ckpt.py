import argparse

import torch

from cs336_basics.BPETokenizer import BPETokenizer
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.util_funcs import decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, default="ckpts/latest.pt")
    p.add_argument(
        "--vocab_path",
        type=str,
        default="/home/xiaodong/ws/cs336/data/bpe_artifacts_TinyStoriesV2-GPT4-train/vocab.hex.json",
    )
    p.add_argument(
        "--merges_path",
        type=str,
        default="/home/xiaodong/ws/cs336/data/bpe_artifacts_TinyStoriesV2-GPT4-train/merges.hex.json",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--eot_token",
        type=str,
        default=None,
        help="Optional stop token. If omitted or missing from vocab, generation uses max_new_tokens only.",
    )

    # If omitted, these are inferred from the checkpoint where possible.
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--theta", type=int, default=10000)
    p.add_argument("--num_layers", "--num_layer", dest="num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    return p.parse_args()


def infer_model_config(model_state: dict[str, torch.Tensor]) -> dict[str, int]:
    if "embed.W" not in model_state:
        raise KeyError("Checkpoint is missing required key 'embed.W'.")

    layer_indices = {
        int(key.split(".")[1])
        for key in model_state
        if key.startswith("transformer_stack.")
    }
    if not layer_indices:
        raise ValueError("Checkpoint does not contain any transformer layers.")

    return {
        "vocab_size": int(model_state["embed.W"].shape[0]),
        "d_model": int(model_state["embed.W"].shape[1]),
        "d_ff": int(model_state["transformer_stack.0.ffn.w1"].shape[0]),
        "num_layers": max(layer_indices) + 1,
    }


def resolve_arg(name: str, explicit_value: int | None, inferred_value: int | None) -> int:
    if explicit_value is None and inferred_value is None:
        raise ValueError(f"Could not determine `{name}` from the checkpoint. Please pass `--{name}`.")
    if explicit_value is not None and inferred_value is not None and explicit_value != inferred_value:
        raise ValueError(
            f"`--{name}={explicit_value}` does not match the checkpoint value `{inferred_value}`."
        )
    return inferred_value if explicit_value is None else explicit_value


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    special_tokens = [args.eot_token] if args.eot_token else None
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=special_tokens,
    )
    eot_id = None
    if args.eot_token:
        eot_id = tokenizer._inv_vocab.get(args.eot_token.encode("utf-8"))
        if eot_id is None:
            print(f"Warning: EOT token not in vocab, ignoring stop token: {args.eot_token}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    inferred_cfg = infer_model_config(ckpt["model"])
    vocab_size = resolve_arg("vocab_size", args.vocab_size, inferred_cfg["vocab_size"])
    d_model = resolve_arg("d_model", args.d_model, inferred_cfg["d_model"])
    d_ff = resolve_arg("d_ff", args.d_ff, inferred_cfg["d_ff"])
    num_layers = resolve_arg("num_layers", args.num_layers, inferred_cfg["num_layers"])
    context_length = resolve_arg("context_length", args.context_length, None)
    num_heads = resolve_arg("num_heads", args.num_heads, None)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=args.theta,
        weight_tying=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    token_ids = tokenizer.encode(args.prompt)
    if len(token_ids) == 0:
        raise ValueError("Prompt tokenized to empty sequence.")

    generated_ids = list(token_ids)
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            x = torch.tensor(
                [generated_ids[-context_length:]],
                dtype=torch.long,
                device=device,
            )
            logits = model(x)
            next_id = decode(
                llm_output=logits,
                temperature=args.temperature,
                prob_threshold=args.top_p,
            )
            generated_ids.append(next_id)
            if eot_id is not None and next_id == eot_id:
                break

    full_text = tokenizer.decode(generated_ids)
    completion_text = tokenizer.decode(generated_ids[len(token_ids) :])

    print("=== PROMPT ===")
    print(args.prompt)
    print("\n=== COMPLETION ===")
    print(completion_text)
    print("\n=== FULL TEXT ===")
    print(full_text)


if __name__ == "__main__":
    main()
