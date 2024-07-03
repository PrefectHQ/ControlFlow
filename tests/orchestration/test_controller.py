from controlflow.flows import Flow
from controlflow.orchestration.controller import Controller
from controlflow.tasks import Task


class TestReadyTasks:
    def test_ready_tasks(self):
        controller = Controller(flow=Flow())
        assert controller.get_ready_tasks() == []

    def test_ready_tasks_nested_1(self):
        with Flow() as flow:
            with Task("parent") as parent:
                child_1 = Task("child 1")
                child_2 = Task("child 2")

        assert Controller(flow=flow, tasks=[]).get_ready_tasks() == [child_1, child_2]
        assert Controller(flow=flow, tasks=[child_1]).get_ready_tasks() == [child_1]
        assert Controller(flow=flow, tasks=[child_2]).get_ready_tasks() == [child_2]
        assert Controller(flow=flow, tasks=[parent]).get_ready_tasks() == [
            child_1,
            child_2,
        ]

    def test_ready_tasks_nested(self):
        with Flow() as flow:
            with Task("parent"):
                child_1 = Task("child 1")
                child_2 = Task("child 2", context=dict(sibling=child_1))

        assert Controller(flow=flow, tasks=[child_2]).get_ready_tasks() == [child_1]
        assert Controller(flow=flow, tasks=[]).get_ready_tasks() == [child_1]
        assert Controller(flow=flow, tasks=[child_1]).get_ready_tasks() == [child_1]
