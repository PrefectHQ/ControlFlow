from pydantic import BaseModel

import controlflow as cf


class CodeExplanation(BaseModel):
    code: str
    explanation: str
    language: str


def explain_code(code: str, language: str = None) -> CodeExplanation:
    return cf.run(
        f"Explain the following code snippet",
        result_type=CodeExplanation,
        context={"code": code, "language": language or "auto-detect"},
    )


if __name__ == "__main__":
    code_snippet = """
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    """

    result = explain_code(code_snippet, "Python")
    print(f"Code:\n{result.code}\n")
    print(f"Explanation:\n{result.explanation}")
