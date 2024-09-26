import controlflow as cf


@cf.flow(args_as_context=False)
def process_user_data(user_name: str, sensitive_info: str):
    # Main flow context
    print(f"Processing data for user: {user_name}")

    # Create a private flow to handle sensitive information
    with cf.Flow() as private_flow:
        # This task runs in an isolated context
        masked_info = cf.run(
            "Mask the sensitive information",
            context={"sensitive_info": sensitive_info},
            result_type=str,
        )

    # Task in the main flow can be provided the masked_info as context
    summary = cf.run(
        "Summarize the data processing result",
        context={"user_name": user_name, "masked_info": masked_info},
        result_type=str,
    )

    return summary


if __name__ == "__main__":
    result = process_user_data("Alice", "SSN: 123-45-6789")
    print(result)
