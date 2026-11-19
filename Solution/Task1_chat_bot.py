"""
Task 1: Console based Chat Bot
Gets input from the user, sends it to Claude, and prints the
conversation (question and answer) in the console.

Setup:
    pip install anthropic python-dotenv
    Create a .env file next to this script with:
        ANTHROPIC_API_KEY=your_key_here
Run:
    python chat_bot.py
"""

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # automatically loading the .env variables

client = Anthropic()
model = "claude-haiku-5-5"


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
    return next(block.text for block in message.content if block.type == "text")


def main():
    messages = []
    print("Chat Bot started. Type 'quit' to exit.\n")

    while True:
        # Step 1: get the input from the user
        user_input = input("You: ").strip()

        # Step 5: quit
        if user_input.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        if not user_input:
            continue

        # Step 2: prepare the message object
        add_user_message(messages, user_input)

        # Step 3: invoke the llm
        answer = chat(messages)
        add_assistant_message(messages, answer)

        # Step 4: print it in the console [question and answer]
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
