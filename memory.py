"""
CONCEPT: Long-term memory
-------------------------
`InMemoryStore` is LangGraph's key/value store for facts that should
survive PAST a single conversation and be shared ACROSS threads — the
opposite of the checkpointer (short-term memory), which is scoped to one
thread_id.

Data is organized as: namespace (a tuple, like a folder path) -> key -> value.
Here we namespace by (user_id, "preferences") so "alice" and "bob" each get
their own isolated set of remembered foods, and those preferences are still
there even if alice starts a brand-new conversation (new thread_id) tomorrow.

We create ONE module-level store instance and import it everywhere
(graph.py, app.py) so they all read/write the same in-memory dictionary.
Swap `InMemoryStore()` for a Postgres-backed store in production — nothing
else in this project would need to change.
"""
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()


def get_preferences(user_id: str) -> list[dict]:
    """Read every remembered food preference for a user."""
    items = store.search((user_id, "preferences"))
    return [item.value for item in items]


def save_preference(user_id: str, food: str, sentiment: str) -> None:
    """Write/overwrite a single preference, keyed by the food name so
    re-stating a preference updates it instead of duplicating it."""
    store.put(
        (user_id, "preferences"),
        food.strip().lower(),
        {"food": food.strip(), "sentiment": sentiment},
    )
