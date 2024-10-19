#import openai

import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Set your API key as an environment variable for security
# You can set it in your shell: export OPENAI_API_KEY='your-a1
# pi-key-here'
# Or uncomment and set it here (not recommended for production code):
# os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize the OpenAI client
os.environ["OPENAI_API_KEY"] = "Open API KEY"
client = OpenAI()

def generate_text(prompt, max_tokens=100):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def chat_with_gpt(messages):
    try:
        response: ChatCompletion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    print("Welcome to the OpenAI GPT-3.5 Demo!")
    print("1. Generate text")
    print("2. Chat with GPT")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        prompt = input("Enter a prompt for text generation: ")
        max_tokens = int(input("Enter max tokens for generation (default 100): ") or 100)
        generated_text = generate_text(prompt, max_tokens)
        if generated_text:
            print("\nGenerated Text:")
            print(generated_text)

    elif choice == '2':
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print("\nChat with GPT (type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            messages.append({"role": "user", "content": user_input})
            response = chat_with_gpt(messages)
            if response:
                print("Assistant:", response)
                messages.append({"role": "assistant", "content": response})
            else:
                print("Failed to get a response. Please try again.")

    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")

if __name__ == "__main__":
    main()