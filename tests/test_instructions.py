from controlflow.instructions import get_instructions, instructions


def test_instructions_context():
    assert get_instructions() == []
    with instructions("abc"):
        assert get_instructions() == ["abc"]
    assert get_instructions() == []


def test_instructions_context_nested():
    assert get_instructions() == []
    with instructions("abc"):
        assert get_instructions() == ["abc"]
        with instructions("def"):
            assert get_instructions() == ["abc", "def"]
        assert get_instructions() == ["abc"]
    assert get_instructions() == []


def test_instructions_context_multiple():
    assert get_instructions() == []
    with instructions("abc", "def", "ghi"):
        assert get_instructions() == ["abc", "def", "ghi"]
    assert get_instructions() == []


def test_instructions_context_empty():
    assert get_instructions() == []
    with instructions():
        assert get_instructions() == []
    assert get_instructions() == []
