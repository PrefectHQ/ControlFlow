from typing import List

from pydantic import BaseModel

import controlflow as cf


class StandardAddress(BaseModel):
    city: str
    state: str
    country: str = "USA"


def standardize_addresses(place_names: List[str]) -> List[StandardAddress]:
    return cf.run(
        "Standardize the given place names into consistent postal addresses",
        result_type=List[StandardAddress],
        context={"place_names": place_names},
    )


if __name__ == "__main__":
    place_names = [
        "NYC",
        "New York, NY",
        "Big Apple",
        "Los Angeles, California",
        "LA",
        "San Fran",
        "The Windy City",
    ]

    standardized_addresses = standardize_addresses(place_names)

    for original, standard in zip(place_names, standardized_addresses):
        print(f"Original: {original}")
        print(f"Standardized: {standard}")
        print()
