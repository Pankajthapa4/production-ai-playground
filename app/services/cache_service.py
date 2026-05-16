import hashlib
from cachetools import TTLCache

# maxsize = max prompts in memory
# ttl = 1 hour

prompt_cache = TTLCache(
    maxsize=100,
    ttl=3600
)


def generate_cache_key(message, system_prompt, model):
    raw = f"{model}:{system_prompt}:{message}"

    return hashlib.sha256(
        raw.encode()
    ).hexdigest()


def get_cached_response(cache_key):
    return prompt_cache.get(cache_key)


def set_cached_response(cache_key, value):
    prompt_cache[cache_key] = value