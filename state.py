"""
CONCEPT: State & Reducers
--------------------------
LangGraph passes a single State object between every node. Each field can
have its own "reducer" — a function that decides HOW new data returned by a
node gets merged into the existing state, instead of just overwriting it.

- `messages` uses the built-in `add_messages` reducer: when a node returns
  {"messages": [new_msg]}, LangGraph APPENDS it to the existing list (and
  also knows how to overwrite a message by matching `id`, which is how tool
  results get slotted next to their tool call). Without a reducer, returning
  {"messages": [...]} would just REPLACE the whole list every time.

- `search_results` uses a small CUSTOM reducer (`append_unique`) that we
  write ourselves, to show that reducers aren't magic — they're just
  `(old_value, new_value) -> merged_value` functions you can define for any
  field, not only messages.
"""
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def append_unique(existing: list[str], new: list[str]) -> list[str]:
    """Custom reducer: append new items, skipping ones we've already seen."""
    existing = existing or []
    merged = list(existing)
    seen = set(existing)
    for item in new or []:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


class State(TypedDict):
    # Short-term memory of the conversation itself. The `add_messages`
    # reducer is what makes this behave like a growing chat transcript.
    messages: Annotated[list[AnyMessage], add_messages]

    # Which "user" this conversation belongs to (used as the long-term
    # memory namespace — see memory.py).
    user_id: str

    # Demonstrates a hand-rolled reducer: every tool call that finds pages
    # appends to this list instead of clobbering earlier results.
    search_results: Annotated[list[str], append_unique]
