import pytest
from controlflow import Task
from pydantic import BaseModel


class Name(BaseModel):
    first: str
    last: str


@pytest.mark.usefixtures("unit_test_instructions")
class TestTaskResults:
    def test_task_int_result(self):
        task = Task("return 3", result_type=int)
        assert task.run() == 3

    def test_task_pydantic_result(self):
        task = Task("the name is John Doe", result_type=Name)
        result = task.run()
        assert isinstance(result, Name)
        assert result == Name(first="John", last="Doe")
