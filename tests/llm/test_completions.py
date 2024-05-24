import controlflow.llm.completions


def test_mock_completion(mock_completion):
    mock_completion.set_response("Hello, world! xyz")
    response = controlflow.llm.completions.completion(messages=[{"content": "Hello"}])
    assert response.last_response().choices[0].message.content == "Hello, world! xyz"


async def test_mock_completion_async(mock_completion_async):
    mock_completion_async.set_response("Hello, world! xyz")
    response = await controlflow.llm.completions.completion_async(
        messages=[{"content": "Hello"}]
    )
    assert response.last_response().choices[0].message.content == "Hello, world! xyz"


def test_mock_completion_stream(mock_completion_stream):
    mock_completion_stream.set_response("Hello, world! xyz")
    response = controlflow.llm.completions._completion_stream(
        messages=[{"content": "Hello"}],
    )
    deltas = []
    for delta, snapshot in response:
        deltas.append(delta)

    assert [d.choices[0].delta.content for d in deltas[:5]] == ["H", "e", "l", "l", "o"]


async def test_mock_completion_stream_async(mock_completion_stream_async):
    mock_completion_stream_async.set_response("Hello, world! xyz")
    response = controlflow.llm.completions._completion_stream_async(
        messages=[{"content": "Hello"}], stream=True
    )
    deltas = []
    async for delta, snapshot in response:
        deltas.append(delta)
    assert [d.choices[0].delta.content for d in deltas[:5]] == ["H", "e", "l", "l", "o"]
