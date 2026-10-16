"""Small Responses API adapter; stdlib HTTP, explicit timeouts, no secret persistence."""

from dataclasses import dataclass
import json
import os
import socket
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, ambiguous: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.ambiguous = ambiguous


@dataclass(frozen=True)
class ProviderReply:
    output: list[dict]
    input_tokens: int = 0
    output_tokens: int = 0


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OpenAIProvider:
    kind = "openai"

    def __init__(self, model: str, api_key: str | None = None, *, opener=None):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key or self._api_key.startswith(("your_", "replace_", "paste_")):
            raise ValueError("Configure OPENAI_API_KEY locally before selecting --provider openai")
        if not model or len(model) > 100:
            raise ValueError("Configure OPENAI_MODEL or pass --model")
        self._opener = opener or urllib.request.build_opener(NoRedirect())

    def prepare(self, input_items: list[dict], instructions: str, tools: list[dict],
                report_schema: dict, max_output_tokens: int) -> dict:
        return {
            "model": self.model, "store": False, "input": input_items,
            "instructions": instructions, "tools": tools, "parallel_tool_calls": False,
            "max_output_tokens": max_output_tokens,
            "include": ["reasoning.encrypted_content"],
            "text": {"format": {"type": "json_schema", "name": "risk_investigation",
                                "strict": True, "schema": report_schema}},
        }

    def respond(self, payload: dict, *, timeout: float) -> ProviderReply:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses", method="POST",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderError("Provider response exceeded the size limit", ambiguous=True)
            value = json.loads(raw)
        except urllib.error.HTTPError as error:
            # Do not persist the raw body; an upstream service might echo request details.
            code = error.code
            error.close()
            if code == 429:
                raise ProviderError("Provider rate limit or quota refusal (HTTP 429)", retryable=True) from None
            raise ProviderError(f"Provider request rejected (HTTP {code})",
                                ambiguous=code >= 500) from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, socket.gaierror):
                raise ProviderError("Provider DNS lookup failed before a request could be sent") from None
            raise ProviderError("Provider connection failed; billing status may be unknown",
                                ambiguous=True) from None
        except (TimeoutError, socket.timeout, OSError):
            raise ProviderError("Provider connection failed; billing status may be unknown",
                                ambiguous=True) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProviderError("Provider returned invalid JSON", ambiguous=True) from None
        if value.get("status") != "completed":
            raise ProviderError("Provider response did not complete; no final conclusion accepted", ambiguous=True)
        output, usage = value.get("output"), value.get("usage")
        if not isinstance(output, list) or not isinstance(usage, dict):
            raise ProviderError("Provider response lacked output or usage information", ambiguous=True)
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            raise ProviderError("Provider returned invalid usage information", ambiguous=True)
        if any(not isinstance(item, dict) or item.get("type") not in ("message", "function_call", "reasoning") for item in output):
            raise ProviderError("Provider returned an unsupported output item", ambiguous=True)
        for item in output:
            if item["type"] == "message":
                content = item.get("content")
                if (not isinstance(content, list) or any(not isinstance(part, dict)
                        or part.get("type") not in ("output_text", "refusal") for part in content)):
                    raise ProviderError("Provider returned invalid message content", ambiguous=True)
            if item["type"] == "function_call" and any(
                not isinstance(item.get(key), str) for key in ("call_id", "name", "arguments")
            ):
                raise ProviderError("Provider returned invalid function call fields", ambiguous=True)
        return ProviderReply(output, input_tokens, output_tokens)
