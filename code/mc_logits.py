import argparse
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_choice_scores(
    context: str,
    choices: List[str],
    model,  # AutoModelForCausalLM
    tokenizer,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Compute per-choice log-probabilities and normalized probabilities
    P(choice | context) for a causal language model.

    context: question/prompt text shared by all choices
    choices: list of candidate answers (strings)
    """

    model.to(device)
    model.eval()

    # Prepare token ids for context and choices
    with torch.no_grad():
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)

        choice_ids_list = [
            tokenizer.encode(choice, add_special_tokens=False) for choice in choices
        ]

        # Build sequences: [context_ids + choice_ids]
        input_ids_list = []
        answer_starts = []
        for choice_ids in choice_ids_list:
            answer_start = len(ctx_ids)
            ids = ctx_ids + choice_ids
            answer_starts.append(answer_start)
            input_ids_list.append(ids)

        # Padding
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            # For GPT2-like models without pad token, use eos as pad
            pad_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

        max_len = max(len(ids) for ids in input_ids_list)
        batch_size = len(input_ids_list)

        input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)

        choice_logprobs = []

        for i in range(batch_size):
            ids = input_ids[i]
            answer_start = answer_starts[i]
            answer_len = len(choice_ids_list[i])
            answer_end = answer_start + answer_len

            # Sum log-probabilities of answer tokens given prior context
            token_logprobs = []
            for pos in range(answer_start, answer_end):
                if pos == 0:
                    # No previous token; skip (should not happen if context is non-empty)
                    continue
                prev_pos = pos - 1
                token_id = ids[pos].item()
                token_lp = log_probs[i, prev_pos, token_id]
                token_logprobs.append(token_lp)

            if token_logprobs:
                total_logprob = torch.stack(token_logprobs).sum().item()
            else:
                total_logprob = float("-inf")

            choice_logprobs.append(total_logprob)

        lp_tensor = torch.tensor(choice_logprobs)
        probs = torch.softmax(lp_tensor, dim=-1).tolist()

        results = []
        for choice, lp, p in zip(choices, choice_logprobs, probs):
            results.append(
                {
                    "choice": choice,
                    "logprob": lp,
                    "probability": p,
                }
            )

        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute logits-derived probabilities for multiple-choice answers "
            "using a causal LLM (Hugging Face Transformers)."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path for AutoModelForCausalLM (default: gpt2)",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question text / shared context.",
    )
    parser.add_argument(
        "--choices",
        type=str,
        nargs="+",
        required=True,
        help="Answer options (space-separated list).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on: cuda or cpu (default: auto)",
    )

    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        # Ensure a pad token exists
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    context = args.question.strip() + "\nAnswer: "

    scores = compute_choice_scores(
        context=context,
        choices=args.choices,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )

    print("\nResults (higher logprob/probability is better):")
    for i, s in enumerate(scores):
        print(
            f"Option {i}: '{s['choice']}'\n"
            f"  logprob: {s['logprob']:.4f}\n"
            f"  probability: {s['probability']:.4f}\n"
        )


if __name__ == "__main__":
    main()
