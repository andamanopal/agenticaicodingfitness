# PYTHON — agent.py — The Core Agent Framework
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = "claude-haiku-4-5"

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return message.content[0].text

#make initial list of messages
messages = []

system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""
messages = [
        {
            "role": "user",
            "content": "what time is it now"
        }
]
model = "claude-sonnet-4-6"

def chat(messages, system=None, temperature=1.0, stop_sequences=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)
    return message.content[0].text

# Low temperature - more predictable
#answer = chat(messages, temperature=0.0)
#print(answer)

# High temperature - more creative
answer = chat(messages, temperature=0.0)
print(answer)