"""
CONCEPT: nothing LangGraph-specific here — just wiring up a free LLM.

Gemini 2.5 Flash via Google AI Studio (https://aistudio.google.com/apikey)
has a free tier (no credit card) and is fast/cheap, which is pleasant for a
learning project where you'll be re-running the graph a lot. Swap this out
for any other langchain chat model (ChatOpenAI, ChatGroq, ChatOllama...) if
you prefer — everything else in this project is provider-agnostic, since it
only relies on `.bind_tools()` / `.invoke()` working the standard way.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY is not set.\n"
        "1) Get a free key at https://aistudio.google.com/apikey\n"
        "2) Copy .env.example to .env and paste it in."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2,
)