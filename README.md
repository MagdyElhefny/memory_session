# 🥦 Healthy Food Buddy — a LangGraph learning project

A small chatbot that searches/scrapes the web for healthy food info and
remembers each user's food preferences — built specifically so every core
LangGraph concept is exercised by one coherent app, with a real UI to click
through instead of just reading code.

## Concepts → where to find them

| Concept | File | What to look at |
|---|---|---|
| **Messages** | `state.py` | `messages: Annotated[list[AnyMessage], ...]` — the running chat transcript |
| **Reducers** | `state.py` | `add_messages` (built-in) + `append_unique` (a hand-written custom reducer for `search_results`) |
| **Short-term memory** | `graph.py` (`MemorySaver`) | Conversation state persists across turns for one `thread_id`. Click "New conversation" in the sidebar and the bot forgets the chat — but not your saved preferences. |
| **Long-term memory** | `memory.py`, `graph.py` (`InMemoryStore`) | Preferences are stored under `(user_id, "preferences")` and survive across *different* threads. Try switching the "User id" in the sidebar. |
| **Human-in-the-loop** | `graph.py` (`human_review_node`, `interrupt()`), `app.py`/`cli.py` (`Command(resume=...)`) | When you say "I like/dislike X", the graph **pauses** and asks you to approve before writing to memory. |
| **Tools** (free, no key) | `tools.py` | `search_web` scrapes DuckDuckGo's results page; `scrape_page` fetches and reads an actual URL — real web scraping, no API key |

## How the graph is wired

```
START -> agent -> (route) -> tools ────┐
                     |                 |
                     +-> human_review  |
                     |     ^           |
                     +-----+-----------+
                     |
                     +-> END
```

`agent` calls the LLM. If it wants to search/scrape, we go to `tools`. If it
detected a food preference statement, we go to `human_review` instead, which
pauses (`interrupt()`) until you click Approve/Reject in the UI, then writes
to long-term memory and loops back to `agent` so it can reply.

## Setup

```bash
cd langgraph-food-assistant
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste a free key from https://console.groq.com (no card needed)
```

## Run it

**Streamlit UI (recommended — this is the actual "interface to try"):**
```bash
streamlit run app.py
```
Then in the browser:
1. Ask something like *"find me a healthy high-protein breakfast recipe"* — watch it call `search_web` then `scrape_page`.
2. Say *"I really like avocados"* — it will pause and ask you to approve saving that.
3. Approve it, then check the sidebar — it's now remembered.
4. Click **New conversation** — the chat clears, but ask *"what do I like?"* and it still knows, because that's long-term memory, not chat history.
5. Change the **User id** to `bob` — a totally separate, empty memory.

**Terminal version (simpler, shows the raw interrupt/resume calls):**
```bash
python cli.py
```

## Notes / things to try changing

- Swap `ChatGroq` in `config.py` for `ChatOpenAI`, `ChatOllama`, etc. — nothing else needs to change, since LangGraph just needs a chat model with `.bind_tools()`.
- Add a second custom reducer (e.g. a running token/cost counter) to `state.py` to practice the pattern.
- Replace `InMemoryStore` in `memory.py` with a Postgres-backed store for real persistence across restarts (same `.put`/`.search` API).
- Make `human_review_node` also let you *edit* the food name before approving (the plumbing for it — `decision.get("food", food)` — is already there; just add a text input in `app.py`).
