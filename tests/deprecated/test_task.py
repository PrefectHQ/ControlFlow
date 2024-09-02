import pytest

import controlflow


def test_user_access():
    with pytest.warns(DeprecationWarning):
        t = controlflow.Task("test", user_access=True)
        assert t.interactive
