"""
CONCEPT: message content isn't always a plain string.
---------------------------------------------------
Most models return `AIMessage.content` as a plain `str`. "Thinking" models
like Gemini 2.5 Flash instead return a LIST of content blocks, e.g.:

    [{"type": "text", "text": "..."}, {"type": "thought_signature", ...}]

The extra blocks (thought signatures, etc.) are how the model keeps track
of its own reasoning across turns when you send the message history back
to it — you should leave them in `state["messages"]` untouched so the next
LLM call still has them. But when you just want to SHOW the reply to a
user, you need to pull out only the text parts.
"""


def extract_text(content) -> str:
    """Return the human-readable text of an AIMessage.content, whether it's
    a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)