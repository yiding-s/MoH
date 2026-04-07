import logging
from typing import Optional
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class OpenAIClient(BaseClient):
    """
    OpenAI-compatible LLM client.

    Hydra _target_: utils.llm_client.openai.OpenAIClient

    Provides prompt() and prompt_batch() matching version_iclr's LanguageModel interface.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        batch_size: int = 5,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature, batch_size, cache_dir=cache_dir)

        if OpenAI is None:
            logger.fatal("Package `openai` is required. Install with: pip install openai")
            raise ImportError("openai")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            stream=False,
        )
        return response.choices
