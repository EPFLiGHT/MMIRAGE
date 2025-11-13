# prompts.py

ASSISTANT_ONLY_MD_PROMPT = """
You will receive a JSON object with an array "assistant_texts".

Task:
- Rewrite each string in "assistant_texts" into clear, structured **Markdown** with visible headings and bullet points.
- Keep ONLY information present in the original text. **Do NOT invent** facts.
- Preserve special/redacted tokens exactly (e.g., <|reserved_special_token_0|>).

Output:
- Return **ONLY** a JSON array of strings, in the SAME ORDER and LENGTH as "assistant_texts".
- No extra prose, no code fences, no objects—just an array of strings.

Markdown guidance (use sections only if present in the source):
- "# Summary" (1–3 sentences)
- "## Objective Findings" (bullet list: size/shape/texture/borders/symmetry/measurements)
- "## Study Description / Modality"
- "## Visible Organs/Structures"
- "## Additional Findings" (fluid, artifacts, masses/lesions, vascular features)
- "## Gray Scale / Doppler Features"
- "## Dynamic Features"
- "## Image Quality / Limitations"
- "## Patient Demographics / Context"
- "## Impression / Conclusion"

Example
Input:
{"assistant_texts": [
  "Free-form paragraph about findings and impression..."
]}

Expected output (array only):
[
  "# Summary\\n\\nConcise overview...\\n\\n## Objective Findings\\n- Point one\\n- Point two\\n\\n## Impression / Conclusion\\n- If explicitly present in the source."
]

Now process this input and return only the JSON array:
{payload}
""".strip()