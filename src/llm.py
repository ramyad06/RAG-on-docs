"""LLM layer: DeepInfra OpenAI-compatible client, prompted call, latency timing."""
from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

from src.config import DEEPINFRA_BASE_URL, LLM_MODEL
from src.prompts import SYSTEM_PROMPT, build_user_message

load_dotenv()


@dataclass(frozen=True)
class Answer:
    text: str
    latency_seconds: float


@functools.lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set. Copy .env.example to .env "
            "and add your key."
        )
    return OpenAI(api_key=key, base_url=DEEPINFRA_BASE_URL)


def _build_messages(question: str, chunks: list[Document]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]


def answer(question: str, chunks: list[Document]) -> Answer:
    client = _get_client()
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=_build_messages(question, chunks),
        temperature=0,
    )
    elapsed = time.perf_counter() - start

    # message.content is Optional[str] in the OpenAI SDK; guard it.
    text = (resp.choices[0].message.content or "").strip()
    return Answer(text=text, latency_seconds=elapsed)


def stream_answer(question: str, chunks: list[Document]) -> Iterator[str]:
    """Yield token deltas as they arrive from the LLM.

    Caller can wrap this in time.perf_counter() to measure end-to-end latency.
    """
    client = _get_client()
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=_build_messages(question, chunks),
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
