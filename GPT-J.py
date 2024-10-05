import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import argparse
import time


def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    start_time = time.time()

    # Check if CUDA is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load the model and tokenizer
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

        print(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return None, None, device


def generate_text(model, tokenizer, prompt, max_length=100, device="cpu"):
    try:
        # Encode the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7
            )

        # Decode and return the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return None

'''
def main():
    parser = argparse.ArgumentParser(description="Generate text using GPT-J")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return

    print(f"Generating text for prompt: '{args.prompt}'")
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length, device)

    if generated_text:
        print("\nGenerated Text:")
        print(generated_text)
    else:
        print("Failed to generate text.")
'''
def main():
    model, tokenizer, device = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return

    prompt = input("Enter your prompt: ")
    max_length = int(input("Enter max length (default 100): ") or 100)

    print(f"Generating text for prompt: '{prompt}'")
    generated_text = generate_text(model, tokenizer, prompt, max_length, device)

    if generated_text:
        print("\nGenerated Text:")
        print(generated_text)
    else:
        print("Failed to generate text.")


if __name__ == "__main__":
    main()