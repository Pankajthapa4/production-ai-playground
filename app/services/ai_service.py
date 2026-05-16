import asyncio
import logging
import time
import json
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.services.redis_service import redis_client
from app.services.cache_service import generate_cache_key
from app.services.memory_service import (
    get_conversation_history,
    save_message
)

from app.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    INPUT_COST_PER_1M_TOKENS,
    OUTPUT_COST_PER_1M_TOKENS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL
)

semaphore = asyncio.Semaphore(3)


def calculate_cost(prompt_tokens: int, completion_tokens: int):
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS

    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8)
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8)
)
async def call_llm(session_id: str, message: str, system_prompt: str):

    cache_key = generate_cache_key(
        message=message,
        system_prompt=system_prompt,
        model=GROQ_MODEL
    )

    cached_response = await redis_client.get(cache_key)

    if cached_response:
        logger.info("REDIS CACHE HIT")

        cached_data = json.loads(cached_response)
        cached_data["from_cache"] = True

        return cached_data

    logger.info("REDIS CACHE MISS")

    start_time = time.perf_counter()

    async with semaphore:

        history = await get_conversation_history(session_id)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(history)

        messages.append({
            "role": "user",
            "content": message
        })

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            ),
            timeout=60
        )

        duration = round(
            time.perf_counter() - start_time,
            3
        )

        reply = response.choices[0].message.content

        await save_message(session_id, "user", message)
        await save_message(session_id, "assistant", reply)

        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        cost = calculate_cost(
            prompt_tokens,
            completion_tokens
        )

        result = {
            "reply": reply,
            "model": GROQ_MODEL,
            "duration_seconds": duration,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": cost,
            "from_cache": False
        }

        await redis_client.set(
            cache_key,
            json.dumps(result),
            ex=3600
        )

        logger.info(
            f"LLM COMPLETED | "
            f"Tokens={total_tokens} | "
            f"Cost=${cost['total_cost']} | "
            f"Duration={duration}s"
        )

        return result