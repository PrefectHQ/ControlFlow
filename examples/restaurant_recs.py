from controlflow import Task, flow
from pydantic import BaseModel


class Restaurant(BaseModel):
    name: str
    description: str


@flow
def restaurant_recs(n: int) -> list[Restaurant]:
    """
    An agentic workflow that asks the user for their location and
    cuisine preference, then recommends n restaurants based on their input.
    """

    # get the user's location
    location = Task("Get a location", user_access=True)

    # get the user's preferred cuisine
    cuisine = Task("Get a preferred cuisine", user_access=True)

    # generate the recommendations from the user's input
    recs = Task(
        f"Recommend {n} restaurants to the user",
        context=dict(location=location, cuisine=cuisine),
        result_type=list[Restaurant],
    )
    return recs


if __name__ == "__main__":
    restaurant_recs(5)
