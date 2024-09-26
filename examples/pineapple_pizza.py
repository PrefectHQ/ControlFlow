import controlflow as cf

optimist = cf.Agent(
    name="Half-full",
    instructions="You are an eternal optimist.",
)
pessimist = cf.Agent(
    name="Half-empty",
    instructions="You are an eternal pessimist.",
)
moderator = cf.Agent(name="Moderator")


@cf.flow
def demo(topic: str):
    cf.run(
        "Have a debate about the topic.",
        instructions="Each agent should take at least two turns.",
        agents=[optimist, pessimist],
        context={"topic": topic},
    )

    winner: cf.Agent = cf.run(
        "Whose argument do you find more compelling?",
        agents=[moderator],
        result_type=[optimist, pessimist],
    )

    print(f"{winner.name} wins the debate!")


if __name__ == "__main__":
    demo("pineapple on pizza")
