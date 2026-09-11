"""
STAGE 1: Plain chatbot - no memory, no data, no tools.
Uses Groq's free API (OpenAI-compatible format).
"""

from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and sets the environment variables from it

# This automatically reads your GROQ_API_KEY environment variable
client = Groq()

def chat(user_message):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # free-tier model on Groq (developer rate limits, no card needed)
        max_tokens=500,
        messages=[
            {"role": "system", "content": "You are a friendly customer support assistant for an online shoe store called StepUp."},
            {"role": "user", "content": user_message}
        ]
    )
    # The reply text is here for Groq's OpenAI-style response format
    return response.choices[0].message.content


if __name__ == "__main__":
    print("StepUp Bot: Hi! Ask me anything (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        reply = chat(user_input)
        print(f"StepUp Bot: {reply}\n")