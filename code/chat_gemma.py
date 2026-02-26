import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
# Ensure you have accepted the license on Hugging Face and logged in via `huggingface-cli login`.
MODEL_ID = "google/gemma-3-1b-it"

def main():
    # Determine the best available device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")

    print(f"Loading model: {MODEL_ID}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        ).to(device)
    except OSError as e:
        print(f"Error loading model. Ensure the model ID is correct and you are logged in via huggingface-cli if required.\n{e}")
        return

    print("Model loaded successfully. Type 'quit' or 'exit' to stop.")
    print("-" * 50)

    chat_history = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # Append user message to history
            chat_history.append({"role": "user", "content": user_input})
            
            # Apply the chat template (handles system prompts and formatting)
            prompt = tokenizer.apply_chat_template(
                chat_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize and move to device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
            
            # Decode response (skipping the input tokens)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"Gemma: {response_text}")
            print("-" * 50)
            
            # Append model response to history for context in next turn
            chat_history.append({"role": "model", "content": response_text})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()