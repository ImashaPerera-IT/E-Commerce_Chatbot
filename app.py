"""
Streamlit UI for the StepUp agent.
Wraps the same agent logic from agent.py in a web chat interface.
"""

import csv
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

# ---------- Same tool functions as agent.py ----------

def check_order_status(order_id):
    with open("data/orders.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["order_id"] == str(order_id):
                return row
    return {"error": f"No order found with ID {order_id}"}


def search_products(query):
    query = query.lower()
    matches = []
    with open("data/products.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if query in row["name"].lower() or query in row["category"].lower():
                matches.append(row)
    return matches if matches else {"message": "No matching products found"}


def generate_chart(chart_type):
    """
    Returns (figure, summary_dict).
    figure is shown directly to the user; summary_dict is what the LLM
    reads to describe the chart in words.
    """
    if chart_type == "order_status":
        df = pd.read_csv("data/orders.csv")
        counts = df["status"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax, color="#113F67")
        ax.set_title("Orders by Status", fontsize=13, fontweight="bold", color="#113F67")
        ax.set_xlabel("Status")
        ax.set_ylabel("Number of Orders")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig, counts.to_dict()

    elif chart_type == "stock_by_category":
        df = pd.read_csv("data/products.csv")
        stock_sum = df.groupby("category")["stock"].sum()
        fig, ax = plt.subplots()
        stock_sum.plot(kind="bar", ax=ax, color="#FF5A36")
        ax.set_title("Total Stock by Category", fontsize=13, fontweight="bold", color="#113F67")
        ax.set_xlabel("Category")
        ax.set_ylabel("Units in Stock")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig, stock_sum.to_dict()

    elif chart_type == "price_range":
        df = pd.read_csv("data/products.csv")
        fig, ax = plt.subplots()
        df["price"].astype(float).plot(kind="hist", ax=ax, bins=5, color="#FFC93C", edgecolor="#113F67")
        ax.set_title("Product Price Distribution", fontsize=13, fontweight="bold", color="#113F67")
        ax.set_xlabel("Price ($)")
        plt.tight_layout()
        summary = {
            "min_price": float(df["price"].astype(float).min()),
            "max_price": float(df["price"].astype(float).max()),
            "avg_price": round(float(df["price"].astype(float).mean()), 2)
        }
        return fig, summary

    else:
        return None, {"error": "Unknown chart_type. Use: order_status, stock_by_category, or price_range"}


available_functions = {
    "check_order_status": check_order_status,
    "search_products": search_products,
    "generate_chart": generate_chart
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Get the status and delivery info for a customer's order using their order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. '1001'"}
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
                    "query": {"type": "string", "description": "Search keyword, e.g. 'running shoes'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": (
                "Generate and display an analytics chart for the store owner/seller. "
                "Use this whenever the user asks for a chart, graph, analysis, or breakdown of data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["order_status", "stock_by_category", "price_range"],
                        "description": (
                            "'order_status' = bar chart of orders grouped by status. "
                            "'stock_by_category' = bar chart of stock units per product category. "
                            "'price_range' = histogram of product prices."
                        )
                    }
                },
                "required": ["chart_type"]
            }
        }
    }
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a friendly assistant for an online shoe store called StepUp, "
        "used by both customers and store staff/sellers. "
        "Use the available tools to look up real order status or search real products before answering. "
        "If asked for a chart, graph, analysis, or breakdown of orders/stock/prices, use the generate_chart tool. "
        "Never make up order details or product info - always use the tools for that."
    )
}

# ---------- Streamlit session state = the bot's "memory" ----------
# st.session_state persists across reruns for THIS user's browser session,
# same idea as conversation_history in the terminal version.

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

# Charts don't survive a rerun unless we store them ourselves - keyed by
# the tool_call_id that generated them, so we can redraw the right one
# next to the right message during history playback.
if "charts" not in st.session_state:
    st.session_state.charts = {}

st.set_page_config(page_title="StepUp Bot", page_icon="👟", layout="centered")

# ---------- Custom styling ----------
# Brand palette: deep navy (#113F67) + vivid orange (#FF5A36) + gold accent (#FFC93C)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background: #FAF7F2;
}

/* Header banner */
.stepup-header {
    background: linear-gradient(135deg, #113F67 0%, #1D5B8F 55%, #FF5A36 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 6px 18px rgba(17, 63, 103, 0.25);
}
.stepup-header h1 {
    font-family: 'Poppins', sans-serif;
    color: #FFFFFF;
    font-size: 30px;
    margin: 0;
}
.stepup-header p {
    color: #FFE8DD;
    margin: 6px 0 0 0;
    font-size: 15px;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 6px;
    margin-bottom: 8px;
}

/* User messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #113F67;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
    color: #FFFFFF;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #FFFFFF;
    border: 1px solid #FFD9C7;
}

/* Tool-call caption badges */
.stCaption {
    color: #FF5A36 !important;
    font-weight: 600;
}

/* Chat input box */
[data-testid="stChatInput"] {
    border: 2px solid #FF5A36;
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #113F67;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar: quick actions + branding ----------
with st.sidebar:
    st.markdown("### 👟 StepUp")
    st.markdown("Your shoe store assistant")
    st.markdown("---")
    st.markdown("**Try asking:**")
    st.markdown("- Status of order 1001\n- Show me running shoes\n- Chart of order statuses\n- Stock by category")

st.markdown("""
<div class="stepup-header">
    <h1>👟 StepUp Support Bot</h1>
    <p>Ask about orders, products, store policies, or request an analytics chart</p>
</div>
""", unsafe_allow_html=True)

# Display past messages (skip the system prompt - user shouldn't see it)
for msg in st.session_state.messages[1:]:
    role = msg["role"] if isinstance(msg, dict) else msg.role

    if role == "tool":
        # If this tool call generated a chart earlier, redraw it here so it
        # reappears in the same spot on every rerun (not just the turn it was made).
        tool_call_id = msg["tool_call_id"] if isinstance(msg, dict) else msg.tool_call_id
        if tool_call_id in st.session_state.charts:
            with st.chat_message("assistant"):
                st.pyplot(st.session_state.charts[tool_call_id])

    elif role in ("user", "assistant"):
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if content:
            with st.chat_message(role):
                st.markdown(content)

# Chat input box at the bottom
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Loop: keep calling the LLM until it responds WITHOUT requesting a tool.
            # This handles both single tool calls and multi-step tool use correctly.
            max_steps = 5
            reply = None

            for _ in range(max_steps):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    max_tokens=500,
                    messages=st.session_state.messages,
                    tools=tools
                )
                response_message = response.choices[0].message

                if not response_message.tool_calls:
                    # Model is done using tools - this is the final answer
                    reply = response_message.content
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    break

                # Model wants to call one or more tools
                st.session_state.messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    st.caption(f"🔧 Calling `{function_name}`...")

                    if function_name == "generate_chart":
                        fig, summary = generate_chart(**function_args)
                        if fig is not None:
                            st.pyplot(fig)  # renders the chart NOW, for this turn
                            # Save it so the history loop can redraw it after future reruns
                            st.session_state.charts[tool_call.id] = fig
                        function_response = summary
                    else:
                        function_response = available_functions[function_name](**function_args)

                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })
                # loop continues - the next iteration sends the tool result back

            if reply is None:
                reply = "Sorry, I had trouble completing that request."
                st.session_state.messages.append({"role": "assistant", "content": reply})

            st.markdown(reply)