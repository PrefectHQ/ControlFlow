from pydantic import BaseModel

import controlflow as cf


class Lesson(BaseModel):
    topic: str
    content: str
    exercises: list[str]


def language_learning_session(language: str) -> None:
    tutor = cf.Agent(
        name="Tutor",
        instructions="""
        You are a friendly and encouraging language tutor. Your goal is to create an 
        engaging and supportive learning environment. Always maintain a warm tone, 
        offer praise for efforts, and provide gentle corrections. Adapt your teaching 
        style to the user's needs and pace. Use casual language to keep the 
        conversation light and fun. When working through exercises:
        - Present one exercise at a time.
        - Provide hints if the user is struggling.
        - Offer the correct answer if the user can't solve it after a few attempts.
        - Use encouraging language throughout the process.
        """,
    )

    @cf.flow(default_agent=tutor)
    def learning_flow():
        user_name = cf.run(
            f"Greet the user, learn their name, and introduce the {language} learning session",
            interactive=True,
            result_type=str,
        )

        print(f"\nWelcome, {user_name}! Let's start your {language} lesson.\n")

        while True:
            lesson = cf.run(
                "Create a fun and engaging language lesson", result_type=Lesson
            )

            print(f"\nToday's topic: {lesson.topic}")
            print(f"Lesson content: {lesson.content}\n")

            for exercise in lesson.exercises:
                print(f"Exercise: {exercise}")
                cf.run(
                    "Work through the exercise with the user",
                    interactive=True,
                    context={"exercise": exercise},
                )

            continue_learning = cf.run(
                "Check if the user wants to continue learning",
                result_type=bool,
                interactive=True,
            )

            if not continue_learning:
                break

        summary = cf.run(
            "Summarize the learning session and provide encouragement",
            context={"user_name": user_name},
            result_type=str,
        )
        print(f"\nSession summary: {summary}")

    learning_flow()


if __name__ == "__main__":
    language = input("Which language would you like to learn? ")
    language_learning_session(language)
