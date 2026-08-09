def process_text(row: dict) -> str:
    """A mock function to simulate processing text."""
    input_text = row.get("text", "")
    return f"Processed: ||| {input_text} |||"
