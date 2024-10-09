from typing import Dict, List

import controlflow as cf

extractor = cf.Agent(
    name="Named Entity Recognizer",
    model="openai/gpt-4o-mini",
)


def extract_entities(text: str) -> List[str]:
    return cf.run(
        "Extract all named entities from the text",
        agents=[extractor],
        result_type=List[str],
        context={"text": text},
    )


def extract_categorized_entities(text: str) -> Dict[str, List[str]]:
    return cf.run(
        "Extract named entities from the text and categorize them",
        instructions="""
        Return a dictionary with the following keys:
        - 'persons': List of person names
        - 'organizations': List of organization names
        - 'locations': List of location names
        - 'dates': List of date references
        - 'events': List of event names
        Only include keys if entities of that type are found in the text.
        """,
        agents=[extractor],
        result_type=Dict[str, List[str]],
        context={"text": text},
    )


if __name__ == "__main__":
    text = "Apple Inc. is planning to open a new store in New York City next month."
    entities = extract_entities(text)
    print("Simple extraction:")
    print(entities)

    text = "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission."
    categorized_entities = extract_categorized_entities(text)
    print("\nCategorized extraction:")
    print(categorized_entities)
