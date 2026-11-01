"""A small client for chat models with tool calling, using the OpenAI-style API that both
Groq (free, in the cloud) and Ollama (on this laptop) speak. Nothing is sent anywhere until
one of them is set up in .env:

    groq_api = gsk_...          free key from console.groq.com
    groq_model = ...            optional; default: the first available of GROQ_MODELS

or, for a model running on this laptop (install Ollama, then e.g. `ollama pull qwen3:4b`):

    ollama_model = qwen3:4b
    ollama_url = http://localhost:11434/v1     optional
"""

import os
from dataclasses import dataclass

import requests
from dotenv import dotenv_values

from .. import ROOT

GROQ_URL = "https://api.groq.com/openai/v1"
OLLAMA_URL = "http://localhost:11434/v1"
# Groq renames and retires models now and then: use the first one it still offers.
GROQ_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3-32b"]


class LLMError(RuntimeError):
    """The language model couldn't be reached or refused the request."""


@dataclass
class LLM:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60

    @property
    def label(self) -> str:
        where = "Groq" if "groq.com" in self.base_url else "this laptop (Ollama)" if "localhost" in self.base_url else self.base_url
        return f"{self.model} on {where}"

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.post(f"{self.base_url}/{path}", json=payload, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMError(f"can't reach {self.base_url}: {e}") from e
        if resp.status_code != 200:
            raise LLMError(f"{resp.status_code} from {self.base_url}: {resp.text[:300]}")
        return resp.json()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One chat turn. Returns the assistant message: {"content": ..., "tool_calls": [...]}."""
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._post("chat/completions", payload)
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected answer from {self.base_url}: {str(data)[:300]}") from e


def _settings(path) -> dict:
    values = {k.strip().lower(): (v or "").strip() for k, v in dotenv_values(path).items()} if path.exists() else {}
    keys = ("groq_api", "groq_model", "ollama_model", "ollama_url")
    return {k: os.environ.get(k.upper()) or values.get(k, "") for k in keys}


def _groq_model(api_key: str) -> str:
    """The first model of GROQ_MODELS that Groq still offers (its list changes over time)."""
    try:
        resp = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        offered = {m["id"] for m in resp.json().get("data", [])}
    except (requests.RequestException, ValueError):
        offered = set()
    return next((m for m in GROQ_MODELS if m in offered), GROQ_MODELS[0])


def describe(path=ROOT / ".env") -> str | None:
    """Which language model .env sets up, without contacting it (for /status)."""
    s = _settings(path)
    if s["groq_api"]:
        return f"{s['groq_model'] or 'auto-picked model'} on Groq"
    if s["ollama_model"]:
        return f"{s['ollama_model']} on this laptop (Ollama)"
    return None


def from_env(path=ROOT / ".env") -> LLM | None:
    """The language model set up in .env, or None (then /ask explains how to set one up)."""
    s = _settings(path)
    if s["groq_api"]:
        return LLM(GROQ_URL, s["groq_model"] or _groq_model(s["groq_api"]), s["groq_api"])
    if s["ollama_model"]:
        return LLM(s["ollama_url"] or OLLAMA_URL, s["ollama_model"], timeout=180)
    return None
