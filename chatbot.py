from __future__ import annotations

from dataclasses import dataclass, field

from rag_pipeline import PipelineResponse, get_pipeline
from utils import logger


@dataclass
class ChatMessage:
    """A single chat message (user or assistant)."""

    role: str  # "user" | "assistant"
    content: str
    sources: list[str] = field(default_factory=list)


class Chatbot:
    """
    Session-scoped chatbot with conversation memory.

    Usage::

        bot = Chatbot()
        response = bot.chat("When does the Python batch start?")
        print(response.answer)
        print(bot.history)  # all messages
    """

    MAX_HISTORY_TURNS: int = 20  # keep last 20 exchanges

    def __init__(self) -> None:
        self.pipeline = get_pipeline()
        self.history: list[ChatMessage] = []

  
    # Public API

    def chat(self, user_message: str) -> ChatMessage:
        """
        Process a user message and return an assistant response.

        Args:
            user_message: The student's question.

        Returns:
            ChatMessage with the assistant's answer and sources.
        """
        # Record user message
        self.history.append(
            ChatMessage(role="user", content=user_message)
        )

        # Get RAG response
        response: PipelineResponse = self.pipeline.ask(user_message)

        # Record assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.answer,
            sources=response.sources,
        )
        self.history.append(assistant_msg)

        # Trim history to prevent unbounded growth
        self._trim_history()

        return assistant_msg

    def clear_history(self) -> None:
        """Clear the entire conversation history."""
        self.history.clear()
        logger.info("Chat history cleared.")

    def get_history_summary(self) -> str:
        """
        Return a condensed summary of recent conversation turns
        for context-aware queries.
        """
        if not self.history:
            return ""

        recent = self.history[-6:]  # last 3 exchanges
        lines: list[str] = []
        for msg in recent:
            prefix = "Student" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def get_last_exchange(self) -> tuple[str, str] | None:
        """Return the last (user_msg, assistant_msg) pair, or None."""
        if len(self.history) < 2:
            return None
        return (
            self.history[-2].content,
            self.history[-1].content,
        )
    # Internals

    def _trim_history(self) -> None:
        """Keep only the last N turns to manage memory."""
        max_msgs = self.MAX_HISTORY_TURNS * 2  # user + assistant per turn
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]



# Module-level singleton

_chatbot: Chatbot | None = None


def get_chatbot() -> Chatbot:
    """Return the singleton Chatbot instance."""
    global _chatbot
    if _chatbot is None:
        _chatbot = Chatbot()
    return _chatbot
