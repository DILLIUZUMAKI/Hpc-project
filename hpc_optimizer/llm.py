"""Small OpenAI-compatible Chat Completions client; secrets never enter stored reports."""
from dataclasses import dataclass, field
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from .sandbox import validate_source


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key: str = field(default="", repr=False)
    timeout: int = 120

    @classmethod
    def from_env(cls):
        return cls(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), os.getenv("LLM_MODEL", ""), os.getenv("LLM_API_KEY", ""))

    def validate(self):
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise LLMError("Use an API base URL without embedded credentials, query or fragment.")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise LLMError("Remote API endpoints must use HTTPS.")
        if not self.model.strip():
            raise LLMError("Set LLM_MODEL to a model enabled by your API provider.")
        if not self.api_key and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise LLMError("Set LLM_API_KEY in your local .env file or the session-only password field.")
        return self


def parse_generation(content):
    if not isinstance(content, str):
        raise LLMError("API response did not contain a text completion.")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise LLMError("Incomplete fenced JSON response.")
        text = "\n".join(lines[1:-1])
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("explanation"), str):
            raise ValueError("Missing explanation")
        code = validate_source(value.get("code"))
    except (ValueError, TypeError) as exc:
        raise LLMError("Model must return valid JSON with code and explanation strings.") from exc
    return {"code": code, "explanation": value["explanation"][:20000]}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class LLMClient:
    def __init__(self, config):
        self.config = config.validate()

    def generate(self, prompt):
        payload = {"model": self.config.model, "messages": [
            {"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        request = urllib.request.Request(self.config.base_url.rstrip("/")+"/chat/completions",
                                         data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            opener = urllib.request.build_opener(NoRedirect())
            with opener.open(request, timeout=self.config.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise LLMError("API response exceeds 2 MB.")
            obj = json.loads(raw)
            choice = obj["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise LLMError("API completion was truncated or filtered; select a suitable model or reduce input.")
            result = parse_generation(choice["message"]["content"])
            # Provider response bodies and request headers are not written to logs.
            result["model"] = self.config.model
            result["usage"] = {k: v for k, v in obj.get("usage", {}).items() if type(v) is int}
            return result
        except urllib.error.HTTPError as exc:
            raise LLMError(f"API returned HTTP {exc.code}. Check your endpoint, model, key and quota.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise LLMError("API connection failed or timed out. Check connectivity and provider availability.") from None
        except (KeyError, IndexError, ValueError, TypeError):
            raise LLMError("API returned an unsupported or malformed Chat Completions response.") from None
