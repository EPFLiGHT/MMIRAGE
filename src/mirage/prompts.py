ASSISTANT_ONLY_MD_PROMPT = """
You will receive a JSON object with a single field "assistant_text".

Task:
- Rewrite "assistant_text" as clearer, well-structured **Markdown**.
- You may add headings, bullet points, and simple formatting to improve readability.
- Keep the original meaning; do **not** invent new facts or interpretations.

Output:
- Return only the rewritten Markdown text as a plain string.
- Do NOT wrap it in JSON, quotes, or code fences.

Input:
{payload}
""".strip()
