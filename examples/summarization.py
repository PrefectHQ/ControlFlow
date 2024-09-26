from pydantic import BaseModel

import controlflow as cf


class Summary(BaseModel):
    summary: str
    key_points: list[str]


def summarize_text(text: str, max_words: int = 100) -> Summary:
    return cf.run(
        f"Summarize the given text in no more than {max_words} words and list key points",
        result_type=Summary,
        context={"text": text},
    )


if __name__ == "__main__":
    long_text = """
    The Internet of Things (IoT) is transforming the way we interact with our
    environment. It refers to the vast network of connected devices that collect
    and share data in real-time. These devices range from simple sensors to
    sophisticated wearables and smart home systems. The IoT has applications in
    various fields, including healthcare, agriculture, and urban planning. In
    healthcare, IoT devices can monitor patients remotely, improving care and
    reducing hospital visits. In agriculture, sensors can track soil moisture and
    crop health, enabling more efficient farming practices. Smart cities use IoT to
    manage traffic, reduce energy consumption, and enhance public safety. However,
    the IoT also raises concerns about data privacy and security, as these
    interconnected devices can be vulnerable to cyber attacks. As the technology
    continues to evolve, addressing these challenges will be crucial for the
    widespread adoption and success of IoT.
    """

    result = summarize_text(long_text)
    print(f"Summary:\n{result.summary}\n")
    print("Key Points:")
    for point in result.key_points:
        print(f"- {point}")
