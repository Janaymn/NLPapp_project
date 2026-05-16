import os
import hashlib
import json
import time
from groq import Groq
from config import SIM_PARAMS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LLMClient:
    """
    Resilient LLM Client utilizing Groq with model switching, caching,
    and exponential back-off to handle rate limits.
    """
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.models = [
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'llama-3.3-70b-versatile'
        ]
        self.current_model_idx = 0
        self.cache = {}

    def _get_cache_key(self, model, messages):
        msg_str = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(f"{model}:{msg_str}".encode()).hexdigest()

    def call(self, messages):
        # 1. Check Cache First
        model = self.models[self.current_model_idx]
        cache_key = self._get_cache_key(model, messages)
        if cache_key in self.cache:
            return self.cache[cache_key]

        max_retries = 5
        retry_delay = 10 # Start with 10s delay

        for attempt in range(max_retries):
            try:
                model = self.models[self.current_model_idx]
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024
                )
                content = resp.choices[0].message.content

                # Update Cache and switch model for next call
                self.cache[self._get_cache_key(model, messages)] = content
                self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
                return content

            except Exception as e:
                # Handle Rate Limit specifically
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit hit for {model}. Attempt {attempt+1}/{max_retries}. Retrying in {retry_delay}s...")

                    # Switch model to try another quota
                    self.current_model_idx = (self.current_model_idx + 1) % len(self.models)

                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential back-off
                else:
                    # For other errors, just raise them
                    print(f"API Error: {e}")
                    raise e

        raise Exception("Max retries reached: Groq API is currently unavailable due to rate limits.")
