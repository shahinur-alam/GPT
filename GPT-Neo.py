from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=100):
    # Load pre-trained model and tokenizer
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
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

# Example usage
prompt = "To be the best AI researcher"
generated_text = generate_text(prompt)
print(generated_text)