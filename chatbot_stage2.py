"""
STAGE 2: Chatbot with memory.
Same as Stage 1, but now we keep a running list of messages
so the bot remembers earlier parts of the conversation.
"""

from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and sets the environment variables from it

client = Groq()

# This list grows every turn - it's the bot's "memory"
# The system message goes in as the first item, same as Stage 1
conversation_history = [
    {"role": "system", "content": "You are a friendly customer support assistant for an online shoe store called StepUp."}
]

def chat(user_message):
    # 1. Add the user's new message to history
    conversation_history.append({"role": "user", "content": user_message})

    # 2. Send the ENTIRE history (not just the new message)
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=500,
        messages=conversation_history
    )

    reply = response.choices[0].message.content

    # 3. Add the bot's reply to history too, so next turn it remembers what IT said
    conversation_history.append({"role": "assistant", "content": reply})

    return reply


if __name__ == "__main__":
    print("StepUp Bot: Hi! Ask me anything (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        reply = chat(user_input)
        print(f"StepUp Bot: {reply}\n")