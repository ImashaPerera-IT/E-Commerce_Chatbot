"""
STAGE 3: RAG chatbot.
Instead of guessing, the bot searches your FAQ data for the most
relevant answer, then uses that as grounding facts before replying.
"""

import csv
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

# ---------- STEP A: Set up the vector database ----------

# Using Chroma's built-in default embedding function (onnxruntime-based).
# This avoids needing sentence-transformers/torch/scikit-learn entirely -
# lighter, faster to install, and works the same way conceptually.
chroma_client = chromadb.Client()  # in-memory vector DB (resets each run)
collection = chroma_client.create_collection(name="faqs")

# ---------- STEP B: Load your FAQ CSV into the vector database ----------

def load_faqs():
    with open("data/faqs.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = []
        answers = []
        ids = []
        for i, row in enumerate(reader):
            questions.append(row["question"])
            answers.append(row["answer"])
            ids.append(str(i))

        # Chroma embeds and stores each question automatically here
        collection.add(
            documents=questions,   # what gets embedded/searched
            metadatas=[{"answer": a} for a in answers],  # the real answer text
            ids=ids
        )
    print(f"Loaded {len(questions)} FAQs into the knowledge base.\n")


# ---------- STEP C: Retrieve the most relevant FAQ for a user question ----------

def retrieve_relevant_faq(user_message, top_k=2):
    results = collection.query(
        query_texts=[user_message],
        n_results=top_k
    )
    # Pull out the matched answers (not just the questions)
    matched_answers = [meta["answer"] for meta in results["metadatas"][0]]
    return matched_answers


# ---------- STEP D: Chat function, now grounded in retrieved facts ----------

conversation_history = [
    {"role": "system", "content": (
        "You are a friendly customer support assistant for an online shoe store called StepUp. "
        "Only answer using the facts provided to you in the 'Relevant info' section below each question. "
        "If the answer isn't in the provided info, say you'll connect them with a human agent."
    )}
]

def chat(user_message):
    # 1. Retrieve relevant facts from the FAQ knowledge base
    relevant_facts = retrieve_relevant_faq(user_message)
    facts_text = "\n".join(f"- {fact}" for fact in relevant_facts)

    # 2. Build a message that includes both the question AND the retrieved facts
    augmented_message = (
        f"Relevant info:\n{facts_text}\n\n"
        f"Customer question: {user_message}"
    )

    conversation_history.append({"role": "user", "content": augmented_message})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=500,
        messages=conversation_history
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    load_faqs()
    print("StepUp Bot (RAG-powered): Hi! Ask me anything (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        reply = chat(user_input)
        print(f"StepUp Bot: {reply}\n")