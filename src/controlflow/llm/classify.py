from typing import Union

import tiktoken
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import TypeAdapter

import controlflow
from controlflow.llm.messages import AIMessage, SystemMessage, UserMessage
from controlflow.llm.models import BaseChatModel


def classify(
    data: str,
    labels: list,
    instructions: str = None,
    context: dict = None,
    model: BaseChatModel = None,
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
            - Label {{ loop.index0 }}: {{ label }}
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
        AIMessage("""
            The best label for the data is Label number
            """),
    ]

    model = model or controlflow.llm.models.get_default_model()

    kwargs = {}
    if isinstance(model, (ChatOpenAI, AzureChatOpenAI)):
        openai_kwargs = _openai_kwargs(model=model, n_labels=len(labels))
        kwargs.update(openai_kwargs)
    else:
        messages.append(
            SystemMessage(
                "Return only the label number, no other information or tokens."
            )
        )

    result = controlflow.llm.completions.completion(
        messages=messages,
        model=model,
        max_tokens=1,
        **kwargs,
    )

    index = int(result[0].content)
    return labels[index]


def _openai_kwargs(model: Union[AzureChatOpenAI, ChatOpenAI], n_labels: int):
    encoding = tiktoken.encoding_for_model(model.model_name)

    logit_bias = {}
    for i in range(n_labels):
        for token in encoding.encode(str(i)):
            logit_bias[token] = 100

    return dict(logit_bias=logit_bias)
