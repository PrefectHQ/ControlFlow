import pytest

import controlflow


# deprecated in 0.9
def test_user_access():
    with pytest.warns(DeprecationWarning):
        a = controlflow.Agent(user_access=True)
        assert a.interactive is True
