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
        task = Task("return 3", result_type=int)
        assert task.run() == 3

    def test_task_pydantic_result(self):
        task = Task("the name is John Doe", result_type=Name)
        result = task.run()
        assert isinstance(result, Name)
        assert result == Name(first="John", last="Doe")

    @pytest.xfail(reason="Need to revisit dataframe handling")
    def test_task_dataframe_result(self):
        task = Task(
            'return a dataframe with column "x" that has values 1 and 2 and column "y" that has values 3 and 4',
            result_type=pd.DataFrame,
        )
        result = task.run()
        assert isinstance(result, pd.DataFrame)
        assert result.equals(pd.DataFrame(data={"x": [1, 2], "y": [3, 4]}))
