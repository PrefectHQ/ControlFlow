import pandas as pd
import pytest
from controlflow import Task
from pydantic import BaseModel


class Name(BaseModel):
    first: str
    last: str


@pytest.mark.usefixtures("unit_test_instructions")
class TestTaskResults:
    def test_task_int_result(self):
        task = Task[int]("return 3")
        assert task.run() == 3

    def test_task_pydantic_result(self):
        task = Task[Name]("the name is John Doe")
        result = task.run()
        assert isinstance(result, Name)
        assert result == Name(first="John", last="Doe")

    def test_task_dataframe_result(self):
        task = Task[pd.DataFrame](
            'return a dataframe with column "x" that has values 1 and 2 and column "y" that has values 3 and 4',
        )
        result = task.run()
        assert isinstance(result, pd.DataFrame)
        assert result.equals(pd.DataFrame(data={"x": [1, 2], "y": [3, 4]}))
