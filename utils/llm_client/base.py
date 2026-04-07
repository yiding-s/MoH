import os
import json
import time
import logging
import concurrent.futures
from typing import Optional
from random import random
from datetime import datetime

logger = logging.getLogger(__name__)
#   lm = hydra.utils.instantiate(cfg.llm_client)                                                                                      
#   lm.prompt(expertise, message, temperature=0.7)                                                                             
#   lm.prompt_batch(expertise, [msg1, msg2, msg3]) 

class BaseClient:
    def __init__(self, model: str, temperature: float = 1.0, batch_size: int = 5,
                 cache_dir: Optional[str] = None) -> None:
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._call_counter = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        raise NotImplementedError

    def _single_call(self, messages: list[dict], temperature: float) -> str:
        """Single API call with retry, returns content string."""
        time.sleep(random() * 0.5)
        for attempt in range(10):
            try:
                choices = self._chat_completion_api(messages, temperature, n=1)
                return choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        logger.error("All retry attempts exhausted")
        return "Error: API call failed after retries"

    def _log_to_cache(self, messages: list[dict], temperature: float, response: str):
        if not self.cache_dir:
            return
        self._call_counter += 1
        record = {
            "id": self._call_counter,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
            "response": response,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{self._call_counter:04d}.json"
        path = os.path.join(self.cache_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def prompt(self, expertise: str, message: str, temperature: Optional[float] = None) -> str:
        """
        Single prompt call
        Args:
            expertise: system role content
            message: user message content
            temperature: sampling temperature (defaults to self.temperature)
        """
        temperature = temperature or self.temperature
        messages = [
            {"role": "system", "content": expertise},
            {"role": "user", "content": message},
        ]
        response = self._single_call(messages, temperature)
        self._log_to_cache(messages, temperature, response)
        return response

    def prompt_batch(self, expertise: str, message_batch: list[str], temperature: Optional[float] = None) -> list[str]:
        """
        Parallel batch prompt call.
        Args:
            expertise: system role content (shared across batch)
            message_batch: list of user messages
            temperature: sampling temperature (defaults to self.temperature)
        """
        temperature = temperature or self.temperature

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(message_batch)) as executor:
            futures = {
                executor.submit(self.prompt, expertise, msg, temperature): i
                for i, msg in enumerate(message_batch)
            }
            results = [None] * len(message_batch)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results
