from pydantic import BaseModel, Field

import controlflow as cf


class AnonymizationResult(BaseModel):
    original: str
    anonymized: str
    replacements: dict[str, str] = Field(
        description=r"The replacements made during anonymization, {original} -> {placeholder}"
    )


def anonymize_text(text: str) -> AnonymizationResult:
    return cf.run(
        "Anonymize the given text by replacing personal information with generic placeholders",
        result_type=AnonymizationResult,
        context={"text": text},
    )


if __name__ == "__main__":
    original_text = "John Doe, born on 05/15/1980, lives at 123 Main St, New York. His email is john.doe@example.com."

    result = anonymize_text(original_text)
    print(f"Original: {result.original}")
    print(f"Anonymized: {result.anonymized}")
    print("Replacements:")
    for original, placeholder in result.replacements.items():
        print(f"  {original} -> {placeholder}")
