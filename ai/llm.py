import ollama

SYSTEM_PROMPT = """
You are BOT1, a calm, intelligent AI assistant integrated into a gesture recognition system.
Keep responses concise, helpful, and professional.
"""

def ask_llm(user_input):
    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    return response["message"]["content"]