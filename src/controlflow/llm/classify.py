from pydantic import TypeAdapter

import controlflow
from controlflow.llm.messages import AssistantMessage, SystemMessage, UserMessage


def classify(
    data: str,
    labels: list,
    instructions: str = None,
    context: dict = None,
    model: str = None,
):
    try:
        label_strings = [TypeAdapter(type(t)).dump_json(t).decode() for t in labels]
    except Exception as exc:
        raise ValueError(f"Unable to cast labels to strings: {exc}")

    messages = [
        SystemMessage(
            """ 
            You are an expert classifier that always maintains as much semantic meaning
            as possible when labeling information. You use inference or deduction whenever
            necessary to understand missing or omitted data. Classify the provided data,
            text, or information as one of the provided labels. For boolean labels,
            consider "truthy" or affirmative inputs to be "true".

            ## Labels
            
            You must classify the data as one of the following labels, which are
            numbered (starting from 0) and provide a brief description. Output
            the label number only.
            
            {% for label in labels %}
            - Label #{{ loop.index0 }}: {{ label }}
            {% endfor %}
            """
        ).render(labels=label_strings),
        UserMessage(
            """
            ## Information to classify
            
            {{ data }}
            
            {% if instructions -%}
            ## Additional instructions
            
            {{ instructions }}
            {% endif %}
            
            {% if context -%}
            ## Additional context
            
            {% for key, value in context.items() -%}
            - {{ key }}: {{ value }}
            
            {% endfor %}
            {% endif %}
            
            """
        ).render(data=data, instructions=instructions, context=context),
        AssistantMessage("""
            The best label for the data is Label #
            """),
    ]

    result = controlflow.llm.completions.completion(
        messages=messages,
        model=model,
        max_tokens=1,
        logit_bias={
            str(encoding): 100
            for i in range(len(labels))
            for encoding in _encoder(model)(str(i))
        },
    )

    index = int(result[0].content)
    return labels[index]


def _encoder(model: str):
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except (KeyError, AttributeError):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    return encoding.encode
