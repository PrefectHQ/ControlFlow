import controlflow as cf

# Create a memory module for user preferences
user_preferences = cf.Memory(
    key="user_preferences", instructions="Store and retrieve user preferences."
)

# Create an agent with access to the memory
agent = cf.Agent(memories=[user_preferences])


# Create a flow to ask for the user's favorite color
@cf.flow
def remember_color():
    return cf.run(
        "Ask the user for their favorite color and store it in memory",
        agents=[agent],
        interactive=True,
    )


# Create a flow to recall the user's favorite color
@cf.flow
def recall_color():
    return cf.run(
        "What is the user's favorite color?",
        agents=[agent],
    )


if __name__ == "__main__":
    print("First flow:")
    remember_color()

    print("\nSecond flow:")
    result = recall_color()
    print(result)
