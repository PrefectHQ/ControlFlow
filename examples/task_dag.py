import controlflow
from controlflow import Task, flow

controlflow.settings.enable_tui = True


@flow
def book_ideas():
    genre = Task("pick a genre")

    ideas = Task(
        "generate three short ideas for a book",
        list[str],
        context=dict(genre=genre),
    )

    abstract = Task[str](
        "pick one idea and write a short abstract",
        context=dict(ideas=ideas, genre=genre),
    )

    title = Task[str](
        "pick a title",
        context=dict(abstract=abstract),
    )

    return dict(genre=genre, ideas=ideas, abstract=abstract, title=title)


if __name__ == "__main__":
    result = book_ideas()
    print(result)
