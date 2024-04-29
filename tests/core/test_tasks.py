from control_flow.core.task import Task, get_tasks
from control_flow.utilities.context import ctx


class TestTaskContext:
    def test_context_open_and_close(self):
        assert ctx.get("tasks") == []
        with Task("a") as ta:
            assert ctx.get("tasks") == [ta]
            with Task("b") as tb:
                assert ctx.get("tasks") == [ta, tb]
            assert ctx.get("tasks") == [ta]
        assert ctx.get("tasks") == []

    def test_get_tasks_function(self):
        # assert get_tasks() == []
        with Task("a") as ta:
            assert get_tasks() == [ta]
            with Task("b") as tb:
                assert get_tasks() == [ta, tb]
            assert get_tasks() == [ta]
        assert get_tasks() == []
