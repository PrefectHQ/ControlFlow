import controlflow as cf

classifier = cf.Agent(model="openai/gpt-4o-mini")


def classify_news(headline: str) -> str:
    return cf.run(
        "Classify the news headline into the most appropriate category",
        agents=[classifier],
        result_type=["Politics", "Technology", "Sports", "Entertainment", "Science"],
        context={"headline": headline},
    )


if __name__ == "__main__":
    headline = "New AI Model Breaks Records in Language Understanding"
    category = classify_news(headline)
    print(f"Headline: {headline}")
    print(f"Category: {category}")

    headline = "Scientists Discover Potentially Habitable Exoplanet"
    category = classify_news(headline)
    print(f"\nHeadline: {headline}")
    print(f"Category: {category}")
