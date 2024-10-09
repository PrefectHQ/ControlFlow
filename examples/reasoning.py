"""
This example implements a reasoning loop that lets a relatively simple model
solve difficult problems.

Here, gpt-4o-mini is used to solve a problem that typically requires o1's
reasoning ability.
"""

import argparse

from pydantic import BaseModel, Field

import controlflow as cf
from controlflow.utilities.general import unwrap


class ReasoningStep(BaseModel):
    explanation: str = Field(
        description="""
            A brief (<5 words) description of what you intend to
            achieve in this step, to display to the user.
            """
    )
    reasoning: str = Field(
        description="A single step of reasoning, not more than 1 or 2 sentences."
    )
    found_validated_solution: bool


REASONING_INSTRUCTIONS = """
    You are working on solving a difficult problem (the `goal`). Based
    on your previous thoughts and the overall goal, please perform **one
    reasoning step** that advances you closer to a solution. Document
    your thought process and any intermediate steps you take.
    
    After marking this task complete for a single step, you will be
    given a new reasoning task to continue working on the problem. The
    loop will continue until you have a valid solution.
    
    Complete the task as soon as you have a valid solution.
    
    **Guidelines**
    
    - You will not be able to brute force a solution exhaustively. You
        must use your reasoning ability to make a plan that lets you make
        progress.
    - Each step should be focused on a specific aspect of the problem,
        either advancing your understanding of the problem or validating a
        solution.
    - You should build on previous steps without repeating them.
    - Since you will iterate your reasoning, you can explore multiple
        approaches in different steps.
    - Use logical and analytical thinking to reason through the problem.
    - Ensure that your solution is valid and meets all requirements.
    - If you find yourself spinning your wheels, take a step back and
        re-evaluate your approach.
"""


@cf.flow
def solve_with_reasoning(goal: str, agent: cf.Agent) -> str:
    while True:
        response: ReasoningStep = cf.run(
            objective="""
            Carefully read the `goal` and analyze the problem.
            
            Produce a single step of reasoning that advances you closer to a solution.
            """,
            instructions=REASONING_INSTRUCTIONS,
            result_type=ReasoningStep,
            agents=[agent],
            context=dict(goal=goal),
            model_kwargs=dict(tool_choice="required"),
        )

        if response.found_validated_solution:
            if cf.run(
                """
                Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                """,
                result_type=bool,
                context=dict(goal=goal),
            ):
                break

    return cf.run(objective=goal, agents=[agent])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a reasoning problem.")
    parser.add_argument("--goal", type=str, help="Custom goal to solve", default=None)
    args = parser.parse_args()

    agent = cf.Agent(name="Definitely not GPT-4o mini", model="openai/gpt-4o-mini")

    # Default goal via https://www.reddit.com/r/singularity/comments/1fggo1e/comment/ln3ymsu/
    default_goal = """
        Using only four instances of the digit 9 and any combination of the following
        mathematical operations: the decimal point, parentheses, addition (+),
        subtraction (-), multiplication (*), division (/), factorial (!), and square
        root (sqrt), create an equation that equals 24. 

        In order to validate your result, you should test that you have followed the rules:

        1. You have used the correct number of variables
        2. You have only used 9s and potentially a leading 0 for a decimal
        3. You have used valid mathematical symbols
        4. Your equation truly equates to 24.
        """

    # Use the provided goal if available, otherwise use the default
    goal = args.goal if args.goal is not None else default_goal
    goal = unwrap(goal)
    print(f"The goal is:\n\n{goal}")

    result = solve_with_reasoning(goal=goal, agent=agent)

    print(f"The solution is:\n\n{result}")
