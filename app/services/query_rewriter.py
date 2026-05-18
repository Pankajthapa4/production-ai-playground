

from app.services.ai_service import client
import asyncio

async def rewrite_query(
     session_id: str,
    query: str,
):

    prompt = f"""
You are an AI retrieval query rewriter.

Your job is to convert user queries into:
- clear
- retrieval optimized
- semantically rich
- enterprise search friendly queries

Rules:
- Keep original meaning
- Expand ambiguous references
- Improve retrieval quality
- Add missing context if obvious
- Return ONLY rewritten query

User Query:
{query}
"""

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()