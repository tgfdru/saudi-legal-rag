"""Answer generation with mandatory citations.

- With ANTHROPIC_API_KEY: an LLM answers ONLY from retrieved articles and cites
  each claim as [n]. If the articles don't cover the question it must say so.
- Without a key: an extractive fallback returns the top articles verbatim, so
  the system still works end-to-end (and never hallucinates).
"""
from __future__ import annotations

from . import config
from .chunker import Chunk

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في الأنظمة السعودية (القضاء الإداري وديوان المظالم).
القواعد الإلزامية:
1. أجب فقط من النصوص النظامية المرفقة. لا تستخدم أي معرفة خارجية.
2. وثّق كل معلومة برقم المصدر بين قوسين مربعين مثل [1] أو [2].
3. اذكر اسم النظام ورقم المادة صراحةً عند الاستشهاد.
4. إذا لم تتضمن النصوص ما يجيب عن السؤال، قل حرفياً: "لا أجد في النصوص المتاحة ما يجيب عن هذا السؤال."
5. لا تُصدر أحكاماً قضائية ولا فتاوى؛ أنت تقدّم معلومة نظامية استرشادية فقط.
6. اكتب بالعربية الفصحى، بإيجاز ووضوح، وابدأ بالجواب المباشر ثم التفصيل."""

DISCLAIMER = "⚠️ معلومة استرشادية من نصوص الأنظمة، وليست استشارة قانونية."


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] {c.law} — {c.article}\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def generate(question: str, chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"answer": "لا أجد في النصوص المتاحة ما يجيب عن هذا السؤال.", "mode": "empty"}

    if config.ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=900,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"النصوص النظامية:\n{build_context(chunks)}\n\nالسؤال: {question}",
                }],
            )
            return {"answer": msg.content[0].text, "mode": "llm"}
        except Exception as e:  # network / auth issues -> degrade gracefully
            fallback = _extractive(chunks)
            fallback["error"] = str(e)
            return fallback
    return _extractive(chunks)


def _extractive(chunks: list[Chunk], n: int = 2) -> dict:
    parts = [f"**{c.law} — {c.article}** [{i}]\n{c.text}" for i, c in enumerate(chunks[:n], 1)]
    return {
        "answer": "أقرب النصوص النظامية ذات الصلة بسؤالك:\n\n" + "\n\n".join(parts),
        "mode": "extractive",
    }
