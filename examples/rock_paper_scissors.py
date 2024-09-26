import controlflow as cf


@cf.flow
def rock_paper_scissors():
    """Play rock, paper, scissors against an AI."""
    play_again = True

    while play_again:
        # Get the user's choice on a private thread
        with cf.Flow():
            user_choice = cf.run(
                "Get the user's choice",
                result_type=["rock", "paper", "scissors"],
                interactive=True,
            )

        # Get the AI's choice on a private thread
        with cf.Flow():
            ai_choice = cf.run(
                "Choose rock, paper, or scissors",
                result_type=["rock", "paper", "scissors"],
            )

        # Report the score and ask if the user wants to play again
        play_again = cf.run(
            "Report the score to the user and see if they want to play again.",
            interactive=True,
            context={"user_choice": user_choice, "ai_choice": ai_choice},
            result_type=bool,
        )


if __name__ == "__main__":
    rock_paper_scissors()
