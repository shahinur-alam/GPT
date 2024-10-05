from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode input text and get model predictions
input_text = "To be the best AI researcher"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate continuation
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
