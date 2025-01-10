import controlflow as cf
from controlflow.memory.async_memory import AsyncMemory

from controlflow.memory.providers.postgres import AsyncPostgresMemory
import asyncio


provider = AsyncPostgresMemory(
    database_url="postgresql+psycopg://postgres:postgres@localhost:5432/database",
    # embedding_dimension=1536,
    # embedding_fn=OpenAIEmbeddings(),
    table_name="vector_db_async",
)

# Create a memory module for user preferences
user_preferences = AsyncMemory(
    key="user_preferences",
    instructions="Store and retrieve user preferences.",
    provider=provider,
)

# Create an agent with access to the memory
agent = cf.Agent(memories=[user_preferences])


# Create a flow to ask for the user's favorite color
@cf.flow
async def remember_pet():
    return await cf.run_async(
        "Ask the user for their favorite animal and store it in memory",
        agents=[agent],
        interactive=True,
    )



# Create a flow to recall the user's favorite color
@cf.flow
async def recall_pet():
    return await cf.run_async(
        "What is the user's favorite animal?",
        agents=[agent],
    )

async def main():
    print("First flow:")
    await remember_pet()

    print("\nSecond flow:")
    result = await recall_pet()
    print(result)
    return result

if __name__ == "__main__":
    asyncio.run(main())
