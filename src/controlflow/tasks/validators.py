import re
from typing import Any, Callable, Optional, Sized, TypeVar

T = TypeVar("T")


def chain(*fns: Callable[[T], T]) -> Callable[[T], T]:
    """
    Chain multiple validator functions together.

    This function takes multiple validator functions and returns a new function
    that applies all the validators in sequence. If any validator in the chain
    raises an exception, it will propagate up and stop the validation process.

    Args:
        *fns: Variable number of validator functions to be chained.

    Returns:
        A function that applies all the validator functions in sequence.

    Example:
        >>> def is_even(x: int) -> int:
        ...     if x % 2 != 0:
        ...         raise ValueError("Value must be even")
        ...     return x
        >>> chained = chain(between(min_value=0, max_value=10), is_even)
        >>> chained(4)  # Returns 4
        >>> chained(5)  # Raises ValueError: Value must be even
        >>> chained(12)  # Raises ValueError: Value must be less than or equal to 10
    """

    def chained_validator(value: T) -> T:
        for fn in fns:
            value = fn(value)
        return value

    return chained_validator


def between(
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> Callable[[Any], Any]:
    """
    Create a validator function for values with optional minimum and maximum.

    Args:
        min_value: The minimum allowed value (inclusive). If None, no minimum is enforced.
        max_value: The maximum allowed value (inclusive). If None, no maximum is enforced.

    Returns:
        A function that validates a value based on the specified constraints.

    Raises:
        ValueError: If the value is less than min_value or greater than max_value.

    Example:
        >>> validator = between(min_value=0, max_value=100)
        >>> validator(50)  # Returns 50
        >>> validator(-1)  # Raises ValueError
        >>> validator(101)  # Raises ValueError
    """

    def validate(value: Any) -> Any:
        if min_value is not None and value < min_value:
            raise ValueError(f"Value must be greater than or equal to {min_value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"Value must be less than or equal to {max_value}")
        return value

    return validate


def has_len(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Callable[[Sized], Sized]:
    """
    Create a validator function for any sized object (e.g., list, str, tuple) with optional
    minimum and maximum lengths.

    Args:
        min_length: The minimum allowed length. If None, no minimum is enforced.
        max_length: The maximum allowed length. If None, no maximum is enforced.

    Returns:
        A function that validates a sized object based on the specified length constraints.

    Raises:
        ValueError: If the object length is less than min_length or greater than max_length.

    Example:
        >>> validator = has_len(min_length=2, max_length=5)
        >>> validator([1, 2])  # Returns [1, 2]
        >>> validator("hello")  # Returns "hello"
        >>> validator((1,))  # Raises ValueError
        >>> validator([1, 2, 3, 4, 5, 6])  # Raises ValueError
    """

    def validate(value: Sized) -> Sized:
        if min_length is not None and len(value) < min_length:
            raise ValueError(f"Length must be at least {min_length}")
        if max_length is not None and len(value) > max_length:
            raise ValueError(f"Length must be at most {max_length}")
        return value

    return validate


def is_email() -> Callable[[str], str]:
    """
    Create a validator function for email addresses.

    This validator checks if the given string matches a basic email format.
    It does not perform a comprehensive check against all possible valid email formats,
    but covers most common cases.

    Returns:
        A function that validates an email address string.

    Raises:
        ValueError: If the string is not a valid email address format.

    Example:
        >>> validator = is_email()
        >>> validator("user@example.com")  # Returns "user@example.com"
        >>> validator("invalid-email")  # Raises ValueError
    """
    email_regex = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    def validate(value: str) -> str:
        if not email_regex.match(value):
            raise ValueError("Invalid email address format")
        return value

    return validate


def is_url() -> Callable[[str], str]:
    """
    Create a validator function for URLs.

    This validator checks if the given string matches a basic URL format.
    It does not perform a comprehensive check against all possible valid URL formats,
    but covers most common cases.

    Returns:
        A function that validates a URL string.

    Raises:
        ValueError: If the string is not a valid URL format.

    Example:
        >>> validator = is_url()
        >>> validator("https://www.example.com")  # Returns "https://www.example.com"
        >>> validator("invalid-url")  # Raises ValueError

    """
    url_regex = re.compile(
        r"^((https?:\/\/)?"  # optional protocol
        r"((([a-z\d]([a-z\d-]*[a-z\d])*)\.)+[a-z]{2,}|"  # domain name
        r"((\d{1,3}\.){3}\d{1,3}))"  # OR ip (v4) address
        r"(\:\d+)?(\/[-a-z\d%_.~+]*)*"  # port and path
        r"(\?[;&a-z\d%_.~+=-]*)?"  # query string
        r"(\#[-a-z\d_]*)?$)",  # fragment locator
        re.IGNORECASE,
    )

    def validate(value: str) -> str:
        if not url_regex.match(value):
            raise ValueError("Invalid URL format")
        return value

    return validate


def has_keys(required_keys: set[str]) -> Callable[[dict], dict]:
    """
    Create a validator function for dictionaries to assert that they have certain keys.

    Args:
        required_keys: A set of keys that must be present in the dictionary.

    Returns:
        A function that validates a dictionary based on the specified required keys.

    Raises:
        ValueError: If the dictionary is missing any of the required keys.

    Example:
        >>> validator = has_keys({"name", "age"})
        >>> validator({"name": "John", "age": 30})  # Returns the dict
        >>> validator({"name": "John"})  # Raises ValueError
    """

    def validate(value: dict) -> dict:
        missing_keys = required_keys - set(value.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
        return value

    return validate
