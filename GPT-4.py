import os
from openai import OpenAI

# Set up your API key
# It's better to set this as an environment variable
os.environ["OPENAI_API_KEY"] = "OpenAI API Key"

# Initialize the client
client = OpenAI()

def generate_content(assistant_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Specify GPT-4 model
            messages=[
                {"role": "system", "content": assistant_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return {
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
assistant_prompt = "You are a helpful AI assistant."
user_prompt = "Explain how to be the best AI researcher"

result = generate_content(assistant_prompt, user_prompt)

if result:
    print("Response:", result["response"])
    print("Tokens used:", result["tokens_used"])