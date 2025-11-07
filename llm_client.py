"""
llm_client.py
Provides a SOLID-compliant interface and concrete implementations for various
LLM providers (DeepSeek, Qwen, Gemini).

Inspired by the Go mcp/client.go example, this module uses a Strategy Pattern
where each provider is a concrete class implementing a common interface.
"""

import asyncio
import logging
import json
import httpx
import os
import time
import abc
from typing import List, Optional, Dict, Any, Protocol
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# --------------------------- Interface (Port) ---------------------------

class ILLMClient(Protocol):
    """
    Interface (Port) for the LLM client.
    The engine.py module will type-hint against this protocol.
    """

    async def call(self, system: str, user: str) -> str:
        """Calls the LLM and returns a raw string response."""
        ...


# ----------------------- Abstract Base Implementation ---------------------

class AbstractBaseClient(ILLMClient, abc.ABC):
    """
    Implements the common retry logic from the Go example.
    Concrete clients must implement _call_once.
    """

    def __init__(self,
                 api_key: str,
                 model: str,
                 timeout: int = 120,
                 max_retries: int = 3,
                 max_tokens: int = 4096):

        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Read max_tokens from env, like the Go example
        env_max_tokens = os.getenv("AI_MAX_TOKENS")
        if env_max_tokens:
            try:
                self._max_tokens = int(env_max_tokens)
                logger.info(f"Using AI_MAX_TOKENS from environment: {self._max_tokens}")
            except ValueError:
                self._max_tokens = max_tokens
                logger.warning(f"Invalid AI_MAX_TOKENS env var. Defaulting to {max_tokens}")
        else:
            self._max_tokens = max_tokens

        # Use a single, persistent async client
        self._http_client = httpx.AsyncClient(timeout=self._timeout)

    def _is_retryable_error(self, err: Exception) -> bool:
        """
        Checks if an HTTP error is retryable (e.g., network, timeout).
        Ported from Go's isRetryableError.
        """
        if isinstance(err, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
            return True

        if isinstance(err, httpx.HTTPStatusError):
            # 5xx errors are server-side and worth retrying
            return err.response.status_code >= 500

        err_str = str(err).lower()
        retryable_strings = [
            "eof", "timeout", "connection reset", "connection refused",
            "temporary failure", "no such host", "stream error"
        ]
        return any(s in err_str for s in retryable_strings)

    async def call(self, system: str, user: str) -> str:
        """
        Implements the retry logic from the Go example's CallWithMessages.
        It calls the provider-specific _call_once method.
        """
        last_err: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            if attempt > 1:
                logger.warning(f"Retrying LLM call (Attempt {attempt}/{self._max_retries})...")

            try:
                result = await self._call_once(system, user)
                if attempt > 1:
                    logger.info("LLM call retry successful.")
                return result

            except Exception as e:
                last_err = e
                logger.error(f"LLM call attempt {attempt} failed: {e}")

                if not self._is_retryable_error(e):
                    logger.error("Non-retryable error. Aborting.")
                    break

                if attempt < self._max_retries:
                    wait_time = (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before next retry...")
                    await asyncio.sleep(wait_time)

        raise Exception(f"LLM call failed after {self._max_retries} attempts: {last_err}") from last_err

    @abc.abstractmethod
    async def _call_once(self, system: str, user: str) -> str:
        """
        Provider-specific implementation for a single API call.
        """
        ...

    async def close(self):
        """Closes the underlying HTTP client."""
        await self._http_client.aclose()


# ------------------- OpenAI-Compatible Implementations --------------------

class _OpenAICompatibleClient(AbstractBaseClient):
    """
    Private base class for clients using the OpenAI /chat/completions format.
    (e.g., DeepSeek, Qwen)
    """

    def __init__(self,
                 api_key: str,
                 model: str,
                 base_url: str,
                 use_full_url: bool = False,
                 **kwargs):

        super().__init__(api_key=api_key, model=model, **kwargs)
        self._base_url = base_url
        self._use_full_url = use_full_url

    async def _call_once(self, system: str, user: str) -> str:
        """
        Implements the OpenAI-compatible API call logic.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        request_body = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.5,  # Lowered for JSON stability
            "max_tokens": self._max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        url = self._base_url
        if not self._use_full_url:
            url = f"{self._base_url.rstrip('/')}/chat/completions"

        logger.info(f"Calling OpenAI-Compatible API: {url} (Model: {self._model})")

        response = await self._http_client.post(url, json=request_body, headers=headers)

        # Raise HTTPStatusError for 4xx/5xx responses
        response.raise_for_status()

        data = response.json()

        if not data.get("choices") or not data["choices"][0].get("message"):
            raise Exception(f"Invalid response structure from API: {data}")

        content = data["choices"][0]["message"].get("content", "")
        if not content:
            raise Exception(f"API returned empty message content: {data}")

        return content


class DeepSeekClient(_OpenAICompatibleClient):
    """
    Concrete client for DeepSeek.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1",
                 **kwargs):
        logger.info(f"Initializing DeepSeekClient: {model} @ {base_url}")
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)


class QwenClient(_OpenAICompatibleClient):
    """
    Concrete client for Alibaba Qwen (Dashscope).
    """

    def __init__(self,
                 api_key: str,
                 model: str = "qwen-max",  # Use the model name from Go example
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 **kwargs):
        logger.info(f"Initializing QwenClient: {model} @ {base_url}")
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)


# ---------------------- Gemini-Specific Implementation ----------------------

class GeminiClient(AbstractBaseClient):
    """
    Concrete client for Google Gemini.
    Uses a different API structure than OpenAI.
    """

    def __init__(self,
                 api_key: str = "",  # Per instructions, default to empty string
                 model: str = "gemini-2.5-flash-preview-09-2025",
                 base_url: str = "https://generativelanguage.googleapis.com",
                 **kwargs):

        logger.info(f"Initializing GeminiClient: {model} @ {base_url}")
        super().__init__(api_key=api_key, model=model, **kwargs)
        self._base_url = base_url

    async def _call_once(self, system: str, user: str) -> str:
        """
        Implements the Gemini-specific API call logic.
        """

        # 1. Build request body
        contents = [{"role": "user", "parts": [{"text": user}]}]

        request_body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self._max_tokens,
                "temperature": 0.5
            }
        }

        if system:
            request_body["systemInstruction"] = {"parts": [{"text": system}]}

        # 2. Build URL and Headers
        # Per instructions, API key is a query param, NOT a Bearer token
        url = (
            f"{self._base_url.rstrip('/')}/v1beta/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

        headers = {
            "Content-Type": "application/json"
        }

        logger.info(f"Calling Gemini API: {self._base_url.rstrip('/')}/v1beta/models/{self._model}")

        # 3. Make request
        response = await self._http_client.post(url, json=request_body, headers=headers)

        # Raise HTTPStatusError for 4xx/5xx responses
        response.raise_for_status()

        data = response.json()

        # 4. Parse response
        if not data.get("candidates"):
            # Check for prompt feedback if no candidates (e.g., safety block)
            feedback = data.get("promptFeedback", {})
            raise Exception(f"Gemini API returned no candidates. Feedback: {feedback}")

        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return content
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse Gemini response structure: {e}. Response: {data}")


# -------------------------- Example Usage ----------------------------------

async def main():
    """
    Demonstrates usage of the different LLM clients.
    Set API keys as environment variables to run.
    """

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Default to ""

    system_prompt = "You are a helpful assistant."
    user_prompt = "Hello! What is 2+2?"

    clients: Dict[str, ILLMClient] = {}

    if DEEPSEEK_API_KEY:
        clients["DeepSeek"] = DeepSeekClient(api_key=DEEPSEEK_API_KEY)
    else:
        logger.warning("DEEPSEEK_API_KEY not set. Skipping DeepSeek test.")

    # Gemini client can be initialized without a key (per instructions)
    clients["Gemini"] = GeminiClient(api_key=GEMINI_API_KEY)

    for name, client in clients.items():
        try:
            logger.info(f"\n--- Testing {name} Client ---")
            response = await client.call(system_prompt, user_prompt)
            logger.info(f"Response from {name}: {response}")
        except Exception as e:
            logger.error(f"Failed to call {name}: {e}")
        finally:
            if hasattr(client, 'close'):
                await client.close()  # Close the underlying httpx client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
