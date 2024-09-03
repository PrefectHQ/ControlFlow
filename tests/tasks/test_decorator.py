import controlflow


class TestDecorator:
    def test_decorator(self):
        @controlflow.task
        def write_poem(topic: str) -> str:
            """write a poem about `topic`"""

        task = write_poem.as_task("AI")
        assert task.name == "write_poem"
        assert task.objective == "write a poem about `topic`"
        assert task.result_type is str

    def test_decorator_can_return_objective(self):
        @controlflow.task
        def write_poem(topic: str) -> str:
            return f"write a poem about {topic}"

        task = write_poem.as_task("AI")
        assert task.objective == "write a poem about AI"

    def test_return_value_is_added_to_objective(self):
        @controlflow.task
        def write_poem(topic: str) -> str:
            """Writes a poem."""
            return f"write a poem about {topic}"

        task = write_poem.as_task("AI")
        assert task.objective == "Writes a poem.\n\nwrite a poem about AI"

    def test_return_annotation(self):
        @controlflow.task
        def generate_tags(text: str) -> list[str]:
            """Generate a list of tags for the given text."""

        task = generate_tags.as_task("Fly me to the moon")
        assert task.result_type == list[str]

    def test_objective_can_be_provided_as_kwarg(self):
        @controlflow.task(objective="Write a poem about `topic`")
        def write_poem(topic: str) -> str:
            """Writes a poem."""

        task = write_poem.as_task("AI")
        assert task.objective == "Write a poem about `topic`"

    def test_run_task(self):
        @controlflow.task
        def extract_fruit(text: str) -> list[str]:
            return "Extract any fruit mentioned in the text; all lowercase"

        result = extract_fruit("I like apples and bananas")
        assert result == ["apples", "bananas"]
