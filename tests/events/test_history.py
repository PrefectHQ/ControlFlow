import threading

import pytest

from controlflow.events.events import UserMessage
from controlflow.events.history import FileHistory
from controlflow.flows import Flow


class TestFileHistory:
    def test_write_to_thread_id_file(self, tmp_path):
        h = FileHistory(base_path=tmp_path)
        event = UserMessage(content="test")
        thread_id = "abc"

        # assert a file called 'abc.json' does not exist in tmp_path
        assert not (tmp_path / f"{thread_id}.json").exists()

        h.add_events(thread_id, [event])

        # assert a file called 'abc.json' exists in tmp_path
        assert (tmp_path / f"{thread_id}.json").exists()

    def test_read_from_thread_id_file(self, tmp_path):
        h1 = FileHistory(base_path=tmp_path)
        h2 = FileHistory(base_path=tmp_path)
        event = UserMessage(content="test")
        thread_id = "abc"

        h1.add_events(thread_id, [event])
        # read with different history object
        assert h2.get_events(thread_id) == [event]

    def test_file_histories_respect_base_path(self, tmp_path):
        h1 = FileHistory(base_path=tmp_path)
        h2 = FileHistory(base_path=tmp_path / "subdir")
        event = UserMessage(content="test")
        thread_id = "abc"

        h1.add_events(thread_id, [event])
        # read with different history object
        assert h2.get_events(thread_id) == []
        assert h1.get_events(thread_id) == [event]

    def test_file_history_creates_dir(self, tmp_path):
        h = FileHistory(base_path=tmp_path / "subdir")
        event = UserMessage(content="test")
        thread_id = "abc"

        h.add_events(thread_id, [event])
        assert (tmp_path / "subdir" / f"{thread_id}.json").exists()

    def test_empty_event_content(self, tmp_path):
        """Test adding an event with empty content"""
        h = FileHistory(base_path=tmp_path)
        event = UserMessage(content="")
        thread_id = "abc"
        h.add_events(thread_id, [event])
        assert h.get_events(thread_id) == [event]

    def test_no_events_added(self, tmp_path):
        """Test no events are added and reading from an empty thread"""
        h = FileHistory(base_path=tmp_path)
        thread_id = "xyz"
        assert h.get_events(thread_id) == []

    def test_invalid_file_access(self, tmp_path):
        """Test for file permission errors or non-writable paths"""
        h = FileHistory(base_path="/invalid-path")
        event = UserMessage(content="test")
        thread_id = "abc"
        with pytest.raises(OSError):
            h.add_events(thread_id, [event])


class TestFileHistoryFlow:
    def test_flow_uses_file_history(self, tmp_path):
        f1 = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))
        f2 = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))
        event = UserMessage(content="test")
        f1.add_events([event])
        assert f2.get_events() == [event]

    def test_flow_sets_thread_id_for_file_history(self, tmp_path):
        f1 = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))
        f2 = Flow(thread_id="xyz", history=FileHistory(base_path=tmp_path))
        f3 = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))

        f1.add_events([UserMessage(content="test")])
        assert len(f1.get_events()) == 1
        assert len(f2.get_events()) == 0
        assert len(f3.get_events()) == 1

    @pytest.mark.skip(reason="Not sure what this supposed to test")
    def test_concurrent_access(self, tmp_path):
        """Test concurrent access to the same thread file"""
        f1 = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))
        f2 = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))
        event1 = UserMessage(content="test1")
        event2 = UserMessage(content="test2")

        def write_events(flow, events):
            for event in events:
                flow.add_events([event])

        thread1 = threading.Thread(target=write_events, args=(f1, [event1]))
        thread2 = threading.Thread(target=write_events, args=(f2, [event2]))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        assert len(FileHistory(base_path=tmp_path).get_events("abc")) == 2

    def test_flow_empty_event_content(self, tmp_path):
        """Test flow handling of empty content events"""
        f = Flow(thread_id="abc", history=FileHistory(base_path=tmp_path))
        event = UserMessage(content="")
        f.add_events([event])
        assert f.get_events() == [event]
