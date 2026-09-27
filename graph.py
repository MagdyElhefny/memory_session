"""
CONCEPT: Putting it all together — the graph
---------------------------------------------
Flow:

        START
          |
          v
       [agent] <---------------------------+
          |                                |
   (route_after_agent)                     |
      /        \\             \\             |
     v          v              v            |
   END       [tools]     [human_review] ----+
              |                             ^
              +-----------------------------+

- agent: calls the LLM (with tools bound + known long-term preferences
  injected into the system prompt). Ends the turn if it produced a plain
  text answer; loops back if it asked to use a tool.
- tools: runs `search_web` / `scrape_page` via LangGraph's prebuilt ToolNode.
- human_review: reached ONLY when the model called `propose_preference`.
  Pauses with `interrupt()` (human-in-the-loop) and, once approved, writes
  to the long-term store (memory.py) directly.

Short-term memory: the whole graph is compiled with a `MemorySaver`
checkpointer, so `messages`/`search_results` persist across turns for a
given thread_id (one browser tab / one conversation) and interrupts survive
a "page refresh" of the app.

Long-term memory: compiled with a shared `store` (memory.py), so
preferences persist across DIFFERENT thread_ids for the same user_id.
"""
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from config import llm
from state import State
from tools import ALL_TOOLS, SEARCH_TOOLS
from memory import get_preferences, save_preference, store

llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are a knowledgeable, friendly nutrition assistant. Your ONLY \
purpose is to help users discover healthy food, recipes, and nutrition information. \
Do not answer questions unrelated to food, nutrition, or healthy eating — politely \
redirect the user back to that topic if they ask about something else.

## Tools

- `search_web`: use this FIRST when the user asks about a recipe, food fact, or \
nutrition topic you're not already certain about. Never invent facts, numbers, \
or recipes from memory alone if a quick search could confirm them.
- `scrape_page`: after `search_web` returns promising URLs, use this to read the \
actual page content before answering. Never answer based on a search snippet alone \
— always scrape the source you're citing.
- `propose_preference`: call this the moment the user clearly states they like or \
dislike a specific food (e.g. "I hate mushrooms", "I love salmon"). Pass the food \
and sentiment ('likes' or 'dislikes').
  - Call `propose_preference` ALONE — never combine it with `search_web` or \
`scrape_page` in the same turn. If the user both states a preference AND asks a \
question in the same message, call `propose_preference` first, then handle the \
question in your NEXT turn.
  - Do not guess at preferences from vague or joking statements — only act on \
clear, direct statements of like/dislike.

## Answering style

- Keep answers short, practical, and food-focused. No filler, no long intros.
- Use bullet points for recipe steps or ingredient lists.
- When citing a source, name it briefly (e.g. "According to Healthline...") rather \
than pasting large blocks of text.
- If you're not confident in a nutrition claim after searching, say so plainly \
instead of guessing.

## Memory

What you already remember about this user:
{memories}

- Use these memories naturally to personalize suggestions (e.g. avoid disliked \
foods, lean into liked ones) without explicitly listing them back at the user \
unless asked.
- If memories conflict with what the user just said, trust the most recent message \
and update via `propose_preference`.
"""


def _format_memories(prefs: list[dict]) -> str:
    if not prefs:
        return "(nothing yet)"
    return "\n".join(f"- {p['sentiment']} {p['food']}" for p in prefs)


def agent_node(state: State) -> dict:
    prefs = get_preferences(state["user_id"])
    system = SystemMessage(content=SYSTEM_PROMPT.format(memories=_format_memories(prefs)))
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def route_after_agent(state: State) -> str:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    if any(call["name"] == "propose_preference" for call in last.tool_calls):
        return "human_review"
    return "tools"


def human_review_node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    last = state["messages"][-1]
    call = next(c for c in last.tool_calls if c["name"] == "propose_preference")
    food = call["args"]["food"]
    sentiment = call["args"]["sentiment"]

    # --- HUMAN-IN-THE-LOOP ---
    # Execution pauses HERE. LangGraph saves state via the checkpointer and
    # returns control to whoever called graph.invoke/stream. The app resumes
    # later with Command(resume=<decision>), and THAT value becomes the
    # return value of interrupt() below, and execution continues right here.
    decision = interrupt(
        {
            "type": "preference_approval",
            "food": food,
            "sentiment": sentiment,
            "message": f"Save that the user {sentiment} '{food}'?",
        }
    )

    if decision.get("approved"):
        final_food = decision.get("food", food)
        final_sentiment = decision.get("sentiment", sentiment)
        save_preference(state["user_id"], final_food, final_sentiment)
        note = f"Saved: you {final_sentiment} {final_food}. I'll remember that."
    else:
        note = "Okay, I won't save that preference."

    # Every tool_call must be answered by exactly one ToolMessage with a
    # matching tool_call_id, or the next LLM call will error.
    return {"messages": [ToolMessage(content=note, tool_call_id=call["id"])]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(SEARCH_TOOLS))
builder.add_node("human_review", human_review_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "human_review": "human_review", END: END}
)
builder.add_edge("tools", "agent")
builder.add_edge("human_review", "agent")

# checkpointer -> short-term memory (per thread_id)
# store       -> long-term memory (per user_id, shared across threads)
graph = builder.compile(checkpointer=MemorySaver(), store=store)
