import pytest

from controlflow.tasks.validators import (
    between,
    chain,
    has_keys,
    has_len,
    is_email,
    is_url,
)


def test_chain():
    def add_one(x):
        return x + 1

    def multiply_by_two(x):
        return x * 2

    chained = chain(add_one, multiply_by_two)
    assert chained(3) == 8  # (3 + 1) * 2 = 8


def test_between():
    validator = between(min_value=0, max_value=10)
    assert validator(5) == 5
    with pytest.raises(ValueError):
        validator(-1)
    with pytest.raises(ValueError):
        validator(11)


def test_has_len():
    validator = has_len(min_length=2, max_length=5)
    assert validator([1, 2]) == [1, 2]
    assert validator("hello") == "hello"
    with pytest.raises(ValueError):
        validator([1])
    with pytest.raises(ValueError):
        validator([1, 2, 3, 4, 5, 6])


def test_is_email():
    validator = is_email()
    assert validator("user@example.com") == "user@example.com"
    with pytest.raises(ValueError):
        validator("not-an-email")


def test_is_url():
    validator = is_url()
    assert validator("https://www.example.com") == "https://www.example.com"
    with pytest.raises(ValueError):
        validator("not-a-url")


def test_has_keys():
    validator = has_keys({"name", "age"})
    assert validator({"name": "John", "age": 30}) == {"name": "John", "age": 30}
    with pytest.raises(ValueError):
        validator({"name": "John"})


def test_validators_with_task():
    from controlflow.tasks.task import Task

    task = Task(
        objective="Get a valid email between 10 and 30 characters",
        result_type=str,
        result_validator=chain(has_len(min_length=10, max_length=30), is_email()),
    )

    assert task.validate_result("user@example.com") == "user@example.com"

    with pytest.raises(ValueError):
        task.validate_result("not-an-email")

    with pytest.raises(ValueError):
        task.validate_result("u@ex.com")

    with pytest.raises(ValueError):
        task.validate_result("very.long.email.address@example.com")
