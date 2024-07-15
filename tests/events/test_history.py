from controlflow.events.events import UserMessage
from controlflow.events.history import FileHistory
from controlflow.flows import Flow


class TestFileHistory:
    def test_write_to_thread_id_file(self, tmpdir):
        h = FileHistory(base_path=tmpdir)
        event = UserMessage(content="test")
        thread_id = "abc"

        # assert a file called 'abc.json' does not exist in tmpdir
        assert not (tmpdir / f"{thread_id}.json").exists()

        h.add_events(thread_id, [event])

        # assert a file called 'abc.json' exists in tmpdir
        assert (tmpdir / f"{thread_id}.json").exists()

    def test_read_from_thread_id_file(self, tmpdir):
        h1 = FileHistory(base_path=tmpdir)
        h2 = FileHistory(base_path=tmpdir)
        event = UserMessage(content="test")
        thread_id = "abc"

        h1.add_events(thread_id, [event])
        # read with different history object
        assert h2.get_events(thread_id) == [event]

    def test_file_histories_respect_base_path(self, tmpdir):
        h1 = FileHistory(base_path=tmpdir)
        h2 = FileHistory(base_path=tmpdir / "subdir")
        event = UserMessage(content="test")
        thread_id = "abc"

        h1.add_events(thread_id, [event])
        # read with different history object
        assert h2.get_events(thread_id) == []
        assert h1.get_events(thread_id) == [event]

    def test_file_history_creates_dir(self, tmpdir):
        h = FileHistory(base_path=tmpdir / "subdir")
        event = UserMessage(content="test")
        thread_id = "abc"

        h.add_events(thread_id, [event])
        assert (tmpdir / "subdir" / f"{thread_id}.json").exists()


class TestFileHistoryFlow:
    def test_flow_uses_file_history(self, tmpdir):
        f1 = Flow(thread_id="abc", history=FileHistory(base_path=tmpdir))
        f2 = Flow(thread_id="abc", history=FileHistory(base_path=tmpdir))
        event = UserMessage(content="test")
        f1.add_events([event])
        assert f2.get_events() == [event]

    def test_flow_sets_thread_id_for_file_history(self, tmpdir):
        f1 = Flow(thread_id="abc", history=FileHistory(base_path=tmpdir))
        f2 = Flow(thread_id="xyz", history=FileHistory(base_path=tmpdir))
        f3 = Flow(thread_id="abc", history=FileHistory(base_path=tmpdir))

        f1.add_events([UserMessage(content="test")])
        assert len(f1.get_events()) == 1
        assert len(f2.get_events()) == 0
        assert len(f3.get_events()) == 1
