import controlflow as cf
from controlflow.tasks.validators import between

optimist = cf.Agent(model="openai/gpt-4o-mini")


def sentiment(text: str) -> float:
    return cf.run(
        "Classify the sentiment of the text as a value between 0 and 1",
        agents=[optimist],
        result_type=float,
        result_validator=between(0, 1),
        context={"text": text},
    )


if __name__ == "__main__":
    print(sentiment("I love ControlFlow!"))

    long_text = """
    Far out in the uncharted backwaters of the unfashionable end of 
    the western spiral arm of the Galaxy lies a small unregarded yellow sun. 
    Orbiting this at a distance of roughly ninety-two million miles is an utterly 
    insignificant little blue-green planet whose ape-descended life forms are so 
    amazingly primitive that they still think digital watches are a pretty neat 
    idea. This planet has – or rather had – a problem, which was this: most of 
    the people living on it were unhappy for pretty much of the time.
    """
    print(sentiment(long_text))
