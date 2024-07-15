from controlflow.utilities.testing import SimpleTask


def test_run_once(default_fake_llm, caplog):
    default_fake_llm.set_responses(["Hello"])
    SimpleTask().run_once()
    assert "run_once is deprecated" in caplog.text


async def test_run_once_async(default_fake_llm, caplog):
    default_fake_llm.set_responses(["Hello"])
    await SimpleTask().run_once_async()
    assert "run_once is deprecated" in caplog.text
