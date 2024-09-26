import random

import controlflow as cf

DEPARTMENTS = [
    "Sales",
    "Support",
    "Billing",
    "Returns",
]


@cf.flow
def routing_flow():
    target_department = random.choice(DEPARTMENTS)

    print(f"\n---\nThe target department is: {target_department}\n---\n")

    customer = cf.Agent(
        name="Customer",
        instructions=f"""
            You are training customer reps by pretending to be a customer
            calling into a call center. You need to be routed to the
            {target_department} department. Come up with a good backstory.
            """,
    )

    trainee = cf.Agent(
        name="Trainee",
        instructions=""",
            You are a trainee customer service representative. You need to
            listen to the customer's story and route them to the correct
            department. Note that the customer is another agent training you.
            """,
    )

    with cf.Task(
        "Route the customer to the correct department.",
        agents=[trainee],
        result_type=DEPARTMENTS,
    ) as main_task:
        while main_task.is_incomplete():
            cf.run(
                "Talk to the trainee.",
                instructions=(
                    "Post a message to talk. In order to help the trainee "
                    "learn, don't be direct about the department you want. "
                    "Instead, share a story that will let them practice. "
                    "After you speak, mark this task as complete."
                ),
                agents=[customer],
                result_type=None,
            )

            cf.run(
                "Talk to the customer.",
                instructions=(
                    "Post a message to talk. Ask questions to learn more "
                    "about the customer. After you speak, mark this task as "
                    "complete. When you have enough information, use the main "
                    "task tool to route the customer to the correct department."
                ),
                agents=[trainee],
                result_type=None,
                tools=[main_task.get_success_tool()],
            )

    if main_task.result == target_department:
        print("Success! The customer was routed to the correct department.")
    else:
        print(
            f"Failed. The customer was routed to the wrong department. "
            f"The correct department was {target_department}."
        )


if __name__ == "__main__":
    routing_flow()
