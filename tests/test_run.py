from controlflow.run import run, run_async


def test_run():
    result = run("what's 2 + 2", result_type=int)
    assert result == 4


async def test_run_async():
    result = await run_async("what's 2 + 2", result_type=int)
    assert result == 4
