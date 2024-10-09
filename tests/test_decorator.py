import asyncio

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

    def test_decorator_can_return_context(self):
        @controlflow.task
        def write_poem(topic: str) -> str:
            return f"write a poem about {topic}"

        task = write_poem.as_task("AI")
        assert task.context["Additional context"] == "write a poem about AI"

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


class TestFlowDecorator:
    def test_sync_flow_decorator(self):
        @controlflow.flow
        def sync_flow():
            return 10

        result = sync_flow()
        assert result == 10

    async def test_async_flow_decorator(self):
        @controlflow.flow
        async def async_flow():
            await asyncio.sleep(0.1)
            return 10

        result = await async_flow()
        assert result == 10

    def test_flow_decorator_preserves_function_metadata(self):
        @controlflow.flow
        def flow_with_metadata():
            """This is a test flow."""
            return 10

        assert flow_with_metadata.__name__ == "flow_with_metadata"
        assert flow_with_metadata.__doc__ == "This is a test flow."

    def test_flow_decorator_with_arguments(self):
        @controlflow.flow(thread="test_thread", instructions="Test instructions")
        def flow_with_args(x: int):
            return x + 10

        result = flow_with_args(5)
        assert result == 15

    async def test_async_flow_decorator_with_arguments(self):
        @controlflow.flow(
            thread="async_test_thread", instructions="Async test instructions"
        )
        async def async_flow_with_args(x: int):
            await asyncio.sleep(0.1)
            return x + 10

        result = await async_flow_with_args(5)
        assert result == 15

    def test_flow_decorator_partial_application(self):
        custom_flow = controlflow.flow(thread="custom_thread")

        @custom_flow
        def partial_flow():
            return 10

        result = partial_flow()
        assert result == 10

    def test_flow_decorator_with_context_kwargs(self):
        @controlflow.flow(context_kwargs=["x", "z"])
        def flow_with_context(x: int, y: int, z: str):
            flow = controlflow.flows.get_flow()
            return flow.context

        result = flow_with_context(1, 2, "test")
        assert result == {"x": 1, "z": "test"}

    def test_flow_decorator_without_context_kwargs(self):
        @controlflow.flow
        def flow_without_context(x: int, y: int, z: str):
            flow = controlflow.flows.get_flow()
            return flow.context

        result = flow_without_context(1, 2, "test")
        assert result == {}

    async def test_async_flow_decorator_with_context_kwargs(self):
        @controlflow.flow(context_kwargs=["a", "b"])
        async def async_flow_with_context(a: int, b: str, c: float):
            flow = controlflow.flows.get_flow()
            return flow.context

        result = await async_flow_with_context(10, "hello", 3.14)
        assert result == {"a": 10, "b": "hello"}


class TestTaskDecorator:
    def test_task_decorator_sync_as_task(self):
        @controlflow.task
        def write_poem(topic: str) -> str:
            """write a two-line poem about `topic`"""

        task = write_poem.as_task("AI")
        assert task.name == "write_poem"
        assert task.objective == "write a two-line poem about `topic`"
        assert task.result_type is str

    def test_task_decorator_async_as_task(self):
        @controlflow.task
        async def write_poem(topic: str) -> str:
            """write a two-line poem about `topic`"""

        task = write_poem.as_task("AI")
        assert task.name == "write_poem"
        assert task.objective == "write a two-line poem about `topic`"
        assert task.result_type is str

    def test_task_decorator_sync(self):
        @controlflow.task
        def write_poem(topic: str) -> str:
            """write a two-line poem about `topic`"""

        assert write_poem("AI")

    async def test_task_decorator_async(self):
        @controlflow.task
        async def write_poem(topic: str) -> str:
            """write a two-line poem about `topic`"""

        assert await write_poem("AI")
