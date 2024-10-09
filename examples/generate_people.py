from pydantic import BaseModel, Field

import controlflow as cf


class UserProfile(BaseModel):
    name: str = Field(description="The full name of the user")
    age: int = Field(description="The age of the user, 20-60")
    occupation: str = Field(description="The occupation of the user")
    hobby: str


def generate_profiles(count: int) -> list[UserProfile]:
    return cf.run(
        f"Generate {count} user profiles",
        result_type=list[UserProfile],
        context={"count": count},
    )


if __name__ == "__main__":
    test_data = generate_profiles(count=5)

    from rich import print

    print(test_data)
