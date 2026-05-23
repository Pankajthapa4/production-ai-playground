import asyncio
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall
)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from app.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL
)


async def run_ragas_evaluation(rows: list[dict]):
    clean_rows = []

    for row in rows:
        clean_rows.append({
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "contexts": [
                str(context)
                for context in row["contexts"]
            ],
              "ground_truth": str(row["ground_truth"])
        })

    dataset = Dataset.from_list(clean_rows)

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            model=GROQ_MODEL,
            temperature=0
        )
    )

    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: evaluate(
            dataset,
            metrics=[
                faithfulness,
                context_precision,
                context_recall
            ],
            llm=evaluator_llm
        )
    )

    return result