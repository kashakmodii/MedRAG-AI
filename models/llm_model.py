"""Local Ollama client for offline text generation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterable

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass(slots=True)
class OllamaResult:
    model: str
    response: str


class OllamaLLM:
    """Talk to a locally running Ollama server over HTTP."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: int = 600) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def _get_available_models(self) -> list[str]:
        """Return model names available in the local Ollama registry."""

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            names: list[str] = []
            for item in models:
                name = item.get("name") if isinstance(item, dict) else None
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
            return names
        except (requests.RequestException, ValueError):
            return []

    def is_running(self) -> bool:
        """Check whether the local Ollama service is reachable."""

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str, models: Iterable[str] = ("mistral", "mistral:latest", "phi", "phi3:mini")) -> OllamaResult:
        """Generate text using the first available model in the fallback list."""

        available_models = set(self._get_available_models())
        candidates = list(models)
        if available_models:
            # Prefer models that are actually installed locally.
            candidates = [model for model in candidates if model in available_models] or list(available_models)

        errors_by_model: list[str] = []
        for model_name in candidates:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            }

            # Retry HTTP generation for transient local timeouts.
            http_error: Exception | None = None
            for _ in range(3):
                try:
                    response = requests.post(self.base_url, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    generated_text = (data.get("response") or "").strip()
                    if generated_text:
                        return OllamaResult(model=model_name, response=generated_text)
                    http_error = RuntimeError(f"Ollama returned an empty response for model '{model_name}'.")
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    http_error = exc

            # Fallback to CLI invocation if HTTP path is unstable.
            try:
                cli_result = subprocess.run(
                    ["ollama", "run", model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
                cli_text = (cli_result.stdout or "").strip()
                if cli_result.returncode == 0 and cli_text:
                    return OllamaResult(model=model_name, response=cli_text)
                if cli_result.stderr:
                    errors_by_model.append(f"{model_name}: CLI error: {cli_result.stderr.strip()}")
                elif http_error is not None:
                    errors_by_model.append(f"{model_name}: HTTP/CLI error: {http_error}")
                else:
                    errors_by_model.append(f"{model_name}: CLI returned empty output")
            except (subprocess.SubprocessError, FileNotFoundError) as exc:
                if http_error is not None:
                    errors_by_model.append(f"{model_name}: HTTP error: {http_error}")
                errors_by_model.append(f"{model_name}: CLI invocation error: {exc}")

        details = " | ".join(errors_by_model) if errors_by_model else "Unknown local runtime issue"
        raise RuntimeError(
            f"Could not generate a response with Ollama. Tried models: {', '.join(candidates)}. Details: {details}"
        )