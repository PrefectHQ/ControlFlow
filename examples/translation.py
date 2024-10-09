from pydantic import BaseModel

import controlflow as cf


class TranslationResult(BaseModel):
    translated: str
    target_language: str


def translate_text(text: str, target_language: str) -> TranslationResult:
    return cf.run(
        f"Translate the given text to {target_language}",
        result_type=TranslationResult,
        context={"text": text, "target_language": target_language},
    )


if __name__ == "__main__":
    original_text = "Hello, how are you?"
    target_language = "French"

    result = translate_text(original_text, target_language)
    print(f"Original: {original_text}")
    print(f"Translated ({result.target_language}): {result.translated}")
