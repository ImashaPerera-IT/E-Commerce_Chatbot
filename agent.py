"""
STAGE 4: Agent with tool calling.
The LLM can now REQUEST that real Python functions run - checking order
status or searching products - instead of just generating text.
"""

import csv
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

# ---------- STEP A: The REAL functions the agent can call ----------

def check_order_status(order_id):
    with open("data/orders.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["order_id"] == str(order_id):
                return row  # returns a dict with status, delivery date, etc.
    return {"error": f"No order found with ID {order_id}"}


def search_products(query):
    query = query.lower()
    matches = []
    with open("data/products.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # simple keyword match against name/category
            if query in row["name"].lower() or query in row["category"].lower():
                matches.append(row)
    return matches if matches else {"message": "No matching products found"}


# A lookup table so we can call the right Python function by name later
available_functions = {
    "check_order_status": check_order_status,
    "search_products": search_products
}

# ---------- STEP B: Describe these functions to the LLM ----------
# This is the "menu" the LLM sees. It does NOT run these - it just knows
# they exist, what they do, and what input each one needs.

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Get the status and delivery info for a customer's order using their order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. '1001'"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog by name or category, e.g. 'running' or 'casual'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword, e.g. 'running shoes' or 'casual'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ---------- STEP C: The agent loop ----------

conversation_history = [
    {"role": "system", "content": (
        "You are a friendly customer support assistant for an online shoe store called StepUp. "
        "Use the available tools to look up real order status or search real products before answering. "
        "Never make up order details or product info - always use the tools for that."
    )}
]

def chat(user_message):
    conversation_history.append({"role": "user", "content": user_message})

    # First call: give the LLM the tools menu and let it decide what to do
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=500,
        messages=conversation_history,
        tools=tools
    )

    response_message = response.choices[0].message

    # Did the LLM ask to call a tool?
    if response_message.tool_calls:
        # Save the LLM's tool request into history
        conversation_history.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  [Agent is calling: {function_name}({function_args})]")

            # Actually run the real Python function
            function_response = available_functions[function_name](**function_args)

            # Send the function's result back to the LLM as a new message
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response)
            })

        # Second call: now the LLM writes a natural reply using the real data
        second_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=500,
            messages=conversation_history
        )
        reply = second_response.choices[0].message.content
    else:
        # No tool needed - just a normal reply
        reply = response_message.content

    conversation_history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    print("StepUp Bot (Agent-powered): Hi! Ask me anything (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        reply = chat(user_input)
        print(f"StepUp Bot: {reply}\n")