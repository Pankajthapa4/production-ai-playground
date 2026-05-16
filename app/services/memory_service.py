import json
from app.services.redis_service import redis_client

MEMORY_TTL_SECONDS = 3600
MAX_MESSAGES = 10


def get_memory_key(session_id: str) -> str:
    return f"chat_memory:{session_id}"


async def get_conversation_history(session_id: str):
    key = get_memory_key(session_id)

    data = await redis_client.get(key)

    if not data:
        return []

    return json.loads(data)


async def save_message(session_id: str, role: str, content: str):
    key = get_memory_key(session_id)

    history = await get_conversation_history(session_id)

    history.append({
        "role": role,
        "content": content
    })

    history = history[-MAX_MESSAGES:]

    await redis_client.set(
        key,
        json.dumps(history),
        ex=MEMORY_TTL_SECONDS
    )


async def clear_conversation_history(session_id: str):
    key = get_memory_key(session_id)
    await redis_client.delete(key)