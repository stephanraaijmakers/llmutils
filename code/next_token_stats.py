# Stephhan Raaijmakers, 2026
import argparse
import math
from typing import Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

# Silence most Transformers warnings (e.g. generation flag notices)
hf_logging.set_verbosity_error()


def select_device(dev: str | None = None) -> str:
    """Select computation device.
    If `explicit` is provided, use it directly. Otherwise prefer CUDA, then MPS,
    falling back to CPU.
    """
    if dev is not None:
        return dev

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, device: str):
    """Load a causal LM and tokenizer on the given device."""

    print(f"Loading model {model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prefer bfloat16/float16 where appropriate, fall back to float32 on CPU
    if device == "cpu":
        dtype = torch.float32
    else:
        # Gemma prefers bfloat16; if unavailable, fall back to float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    return model, tokenizer


def distribution_statistics(probs: np.ndarray) -> Tuple[float, float, float]:
    """Entropy (bits), skewness, and kurtosis of the probability vector.

    - Entropy is computed over the top-N probabilities only.
    - Skewness and kurtosis are computed over the values in `probs` treated as
      a finite sample. Kurtosis here is the standard moment-based value
      E[(X - mu)^4] / E[(X - mu)^2]^2 (not excess kurtosis).
    """

    eps = 1e-20  # small constant to prevent log(0) and division by zero
    p = probs.astype(float)

    # We measure entropy in bits
    entropy_nats = -np.sum(p * np.log(p + eps))
    entropy_bits = entropy_nats / math.log(2.0)

    # Central moments for skewness and kurtosis
    mean = p.mean()
    centered = p - mean
    m2 = np.mean(centered ** 2)

    if m2 == 0.0:
        # All probabilities identical means:  zero skewness, minimal kurtosis (=1)
        return float(entropy_bits), 0.0, 1.0

    m3 = np.mean(centered ** 3)
    m4 = np.mean(centered ** 4)

    std = math.sqrt(m2)
    skewness = m3 / (std ** 3)
    kurtosis = m4 / (m2 ** 2)

    return float(entropy_bits), float(skewness), float(kurtosis)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "[Machine translation] Compute top-N next-token probabilities and logits, along with "
            "entropy, skewness, and kurtosis of the top-N distribution."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it", # change to your preferred causal LM
        help=(
            "Hugging Face model ID (default: google/gemma-3-1b-it). "
            "Must be a causal language model."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt / context to condition on.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top next-token candidates to report (default: 10).",
    )
    parser.add_argument(
        "--completion_tokens",
        type=int,
        default=0,
        help=(
            "Number of tokens to generate as completion and analyze "
            "(approx. words; 0 = no completion)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Computation device: cuda, mps, or cpu. "
            "If omitted, selects automatically."
        ),
    )

    args = parser.parse_args()

    device = select_device(args.device)

    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Tokenize prompt and move to device
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (B, T, V)
    input_ids = inputs["input_ids"]  # (B, T)

    vocab_size = logits.shape[-1]
    k = min(args.top_n, vocab_size)

    # ------------------------------------------------------------------
    # 1. Statistics for tokens that make up the prompt
    # ------------------------------------------------------------------
    seq_len = input_ids.shape[1]
    if seq_len > 1:
        # For perplexity over the prompt tokens we actually predict
        prompt_neg_log_prob_sum = 0.0
        prompt_token_count = 0

        print("# Prompt token statistics (model's prob of each prompt token)")
        print(
            "position,token,token_id,probability,logit,rank,"
            "entropy_bits,skewness,kurtosis,incremental_ppl"
        )

        for pos in range(seq_len - 1):
            # Token being predicted at this step
            token_id = int(input_ids[0, pos + 1])
            token_str = tokenizer.decode([token_id]).replace("\n", "\\n")

            # Distribution predicting this token given previous context
            step_logits = logits[0, pos, :]
            step_probs = torch.softmax(step_logits, dim=-1)

            # Top-N stats for this distribution
            top_probs_step, _ = torch.topk(step_probs, k=k, dim=-1)
            entropy_bits, skewness, kurtosis = distribution_statistics(
                top_probs_step.detach().cpu().numpy()
            )

            prob = float(step_probs[token_id])
            logit = float(step_logits[token_id])
            # Rank of the actual token among all vocabulary items
            rank = int((step_probs > prob).sum().item()) + 1

            # Accumulate for prompt perplexity (natural-log cross-entropy)
            prompt_neg_log_prob_sum += -math.log(prob + 1e-20)
            prompt_token_count += 1

            # Incremental perplexity up to this prompt token
            incremental_ppl = math.exp(
                prompt_neg_log_prob_sum / prompt_token_count
            )

            # position is 1-based index of the token in the prompt
            position = pos + 2

            print(
                f"{position},"
                f"{token_str},"
                f"{token_id},"
                f"{prob:.8f},"
                f"{logit:.8f},"
                f"{rank},"
                f"{entropy_bits:.8f},"
                f"{skewness:.8f},"
                f"{kurtosis:.8f},"
                f"{incremental_ppl:.4f}"
            )

        if prompt_token_count > 0:
            prompt_ppl = math.exp(prompt_neg_log_prob_sum / prompt_token_count)
            print(
                f"# Prompt perplexity over {prompt_token_count} predicted tokens: "
                f"{prompt_ppl:.4f}"
            )

        print()

    # ------------------------------------------------------------------
    # 2. Optional: generate and analyze completion tokens
    # ------------------------------------------------------------------
    if args.completion_tokens > 0:
        print(
            "# Completion token statistics (model's prob of its generated tokens)"
        )
        print(
            "position,token,token_id,probability,logit,rank,"
            "entropy_bits,skewness,kurtosis,incremental_ppl"
        )

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.completion_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Only analyze the newly generated tokens
            orig_len = input_ids.shape[1]
            total_len = generated_ids.shape[1]
            new_tokens = max(total_len - orig_len, 0)

            if new_tokens > 0:
                gen_outputs = model(input_ids=generated_ids)
                gen_logits = gen_outputs.logits

                completion_neg_log_prob_sum = 0.0
                completion_token_count = 0

                for i in range(new_tokens):
                    token_index = orig_len + i
                    prev_index = token_index - 1

                    token_id = int(generated_ids[0, token_index])
                    token_str = tokenizer.decode([token_id]).replace(
                        "\n", "\\n"
                    )

                    step_logits = gen_logits[0, prev_index, :]
                    step_probs = torch.softmax(step_logits, dim=-1)

                    top_probs_step, _ = torch.topk(step_probs, k=k, dim=-1)
                    entropy_bits, skewness, kurtosis = distribution_statistics(
                        top_probs_step.detach().cpu().numpy()
                    )

                    prob = float(step_probs[token_id])
                    logit = float(step_logits[token_id])
                    rank = int((step_probs > prob).sum().item()) + 1

                    # Accumulate for completion perplexity
                    completion_neg_log_prob_sum += -math.log(prob + 1e-20)
                    completion_token_count += 1

                    # Incremental perplexity up to this completion token
                    completion_incremental_ppl = math.exp(
                        completion_neg_log_prob_sum / completion_token_count
                    )

                    # position is 1-based index within the completion
                    position = i + 1

                    print(
                        f"{position},"
                        f"{token_str},"
                        f"{token_id},"
                        f"{prob:.8f},"
                        f"{logit:.8f},"
                        f"{rank},"
                        f"{entropy_bits:.8f},"
                        f"{skewness:.8f},"
                        f"{kurtosis:.8f},"
                        f"{completion_incremental_ppl:.4f}"
                    )

                if completion_token_count > 0:
                    completion_ppl = math.exp(
                        completion_neg_log_prob_sum / completion_token_count
                    )
                    print(
                        "# Completion perplexity over "
                        f"{completion_token_count} generated tokens: "
                        f"{completion_ppl:.4f}"
                    )

                print()

    # ------------------------------------------------------------------
    # 3. top-N next-token distribution at end of prompt
    # ------------------------------------------------------------------
    # Use the last position's logits for next-token distribution
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
    top_logits = last_logits[top_indices]

    # Convert to CPU/NumPy for statistics and printing
    top_probs_np = top_probs.detach().cpu().numpy()
    top_logits_np = top_logits.detach().cpu().numpy()
    top_indices_np = top_indices.detach().cpu().numpy()

    entropy_bits, skewness, kurtosis = distribution_statistics(top_probs_np)

    print("# Next-token top-N distribution at end of prompt")
    # Print CSV-style header
    print(
        "rank,token,token_id,probability,logit,entropy_bits,skewness,kurtosis"
    )

    for rank, (tok_id, prob, logit) in enumerate(
        zip(top_indices_np, top_probs_np, top_logits_np), start=1
    ):
        token_str = tokenizer.decode([int(tok_id)])
        # Replace newlines to keep CSV single-line per token
        token_str = token_str.replace("\n", "\\n")

        print(
            f"{rank},"
            f"{token_str},"
            f"{int(tok_id)},"
            f"{prob:.8f},"
            f"{float(logit):.8f},"
            f"{entropy_bits:.8f},"
            f"{skewness:.8f},"
            f"{kurtosis:.8f}"
        )


if __name__ == "__main__":
    main()
