# prompts.py

ASSISTANT_MD_ENHANCE_PROMPT = """
You will receive a JSON array called a *conversation*, where each element is an object with:
  {{"role": "user" | "assistant", "content": "<string>"}}

Your task:
- **Do not change** the number of messages, their order, or any keys.
- **Copy every `user` message exactly** (byte-for-byte) as-is.
- **Modify only `assistant` messages** by rewriting their `content` into clear, structured **Markdown**.
- Keep **only** information explicitly present in the original assistant content. **Do not invent** new facts.
- **Preserve any redacted or special tokens** (e.g., `<|reserved_special_token_0|>`) exactly as they appear.
- The output must be a **valid JSON array** with the **same structure** as the input, containing only `role` and `content` for each turn.
- No extra commentary, no wrappers, no additional keys.

Markdown guidance for assistant messages:
- Use headings and bullet points to organize the content (e.g., #, ##).
- Prefer concise phrasing; keep factual statements as facts.
- If present in the original assistant content, include sections such as:
  - **Study Description / Modality**
  - **Visible Organs/Structures**
  - **Objective Findings** (size/shape/texture/borders/symmetry/etc.)
  - **Additional Findings** (fluid, artifacts, masses/lesions, vascular features)
  - **Gray Scale / Doppler Features**
  - **Dynamic Features**
  - **Image Quality / Limitations**
  - **Patient Demographics/Context** (only if explicitly stated)
  - **Impression / Conclusion** (only if explicitly stated)
- If some sections are absent in the original text, **omit them** (do not fabricate).

Return **only** the transformed conversation JSON array.

Input conversation:
{conversation_json}
""".strip()
