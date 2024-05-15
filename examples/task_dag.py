from controlflow import Task, flow


@flow
def book_ideas():
    genre = Task("pick a genre", str)
    ideas = Task(
        "generate three short ideas for a book",
        list[str],
        context=dict(genre=genre),
    )
    abstract = Task(
        "pick one idea and write an abstract",
        str,
        context=dict(ideas=ideas, genre=genre),
    )
    title = Task(
        "pick a title",
        str,
        context=dict(abstract=abstract),
    )

    return dict(genre=genre, ideas=ideas, abstract=abstract, title=title)


if __name__ == "__main__":
    result = book_ideas()
    print(result)
