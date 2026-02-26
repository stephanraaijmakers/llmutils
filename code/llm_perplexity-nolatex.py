import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def calculate_surprisal(sentence, model_id="gpt2"):
    # 1. Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode(tid) for tid in input_ids[0]]

    # 3. Get model predictions (logits)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    # 4. Shift logits and labels to align prediction with target
    # Logits at index i predict the token at index i+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # 5. Calculate Cross-Entropy Loss (Log-Likelihood) per token
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # 6. Convert to Surprisal (bits) using log base 2
    surprisals = (token_losses / np.log(2)).tolist()

    print(f"{'Context':<30} | {'Target':<10} | {'Surprisal (bits)':<10}")
    print("-" * 60)
    for i in range(len(surprisals)):
        context = "".join(tokens[:i+1])
        print(f"{context:<30} | {tokens[i+1]:<10} | {surprisals[i]:.2f}")

# Run the analysis
calculate_surprisal("The horse raced past the barn fell")
calculate_surprisal("The cotton clothing is made of grows in Mississippi")
