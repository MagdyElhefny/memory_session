"""
Terminal interface. Simpler than app.py — good for seeing exactly what
interrupt()/Command(resume=...) look like without any UI framework in the way.

Run: python cli.py
"""
import uuid
from langgraph.types import Command
from graph import graph
from memory import get_preferences
from util import extract_text

print("Healthy Food Buddy (CLI) — type 'quit' to exit\n")

user_id = input("User id (e.g. alice): ").strip() or "alice"
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print(f"\nRemembered preferences for '{user_id}':")
for p in get_preferences(user_id):
    print(f"  - {p['sentiment']} {p['food']}")
print()

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit"}:
        break

    result = graph.invoke(
        {"messages": [("user", user_input)], "user_id": user_id, "search_results": []},
        config=config,
    )

    # A run can pause on a human-in-the-loop interrupt any number of times
    # before finally finishing — loop until there's no more interrupt.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[HUMAN REVIEW NEEDED] {payload['message']}")
        answer = input("Approve? (y/n): ").strip().lower()
        result = graph.invoke(
            Command(resume={"approved": answer == "y"}),
            config=config,
        )

    last_message = result["messages"][-1]
    print(f"Bot: {extract_text(last_message.content)}\n")