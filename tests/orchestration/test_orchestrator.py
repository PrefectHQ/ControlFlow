from controlflow.flows import Flow
from controlflow.orchestration.orchestrator import Orchestrator
from controlflow.tasks import Task


class TestReadyTasks:
    def test_ready_tasks(self):
        o = Orchestrator(flow=Flow())
        assert o.get_ready_tasks() == []

    def test_ready_tasks_nested_1(self):
        with Flow() as flow:
            with Task("parent") as parent:
                child_1 = Task("child 1")
                child_2 = Task("child 2")

        assert Orchestrator(flow=flow, tasks=[]).get_ready_tasks() == [child_1, child_2]
        assert Orchestrator(flow=flow, tasks=[child_1]).get_ready_tasks() == [child_1]
        assert Orchestrator(flow=flow, tasks=[child_2]).get_ready_tasks() == [child_2]
        assert Orchestrator(flow=flow, tasks=[parent]).get_ready_tasks() == [
            child_1,
            child_2,
        ]

    def test_ready_tasks_nested(self):
        with Flow() as flow:
            with Task("parent"):
                child_1 = Task("child 1")
                child_2 = Task("child 2", context=dict(sibling=child_1))

        assert Orchestrator(flow=flow, tasks=[child_2]).get_ready_tasks() == [child_1]
        assert Orchestrator(flow=flow, tasks=[]).get_ready_tasks() == [child_1]
        assert Orchestrator(flow=flow, tasks=[child_1]).get_ready_tasks() == [child_1]


class TestMaxIteration:
    def test_max_iteration(self, default_fake_llm, caplog):
        default_fake_llm.set_responses(["Hi."])

        with Flow() as flow:
            t1 = Task("pick a number", max_iterations=2)

        o = Orchestrator(flow=flow, tasks=[t1])

        o.run(steps=2)
        assert t1.is_ready()
        assert "exceeded max iterations" not in caplog.text
        o.run(steps=1)
        assert "exceeded max iterations" in caplog.text
        assert t1.is_failed()
