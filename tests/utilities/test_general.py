import controlflow.utilities.general as general


class TestUnwrap:
    def test_unwrap(self):
        assert general.unwrap("Hello, world!") == "Hello, world!"
        assert (
            general.unwrap("Hello, world!\nThis is a test.")
            == "Hello, world! This is a test."
        )
        assert (
            general.unwrap("Hello, world!\nThis is a test.\n\nThis is another test.")
            == "Hello, world! This is a test.\n\nThis is another test."
        )

    def test_unwrap_with_empty_string(self):
        assert general.unwrap("") == ""

    def test_unwrap_with_multiple_newlines(self):
        assert general.unwrap("\n\n\n") == ""

    def test_unwrap_with_multiline_string(self):
        assert (
            general.unwrap("""
            Hello, world!
            This is a test.
            This is another test.
        """)
            == "Hello, world! This is a test. This is another test."
        )

    def test_unwrap_with_multiline_string_and_newlines(self):
        assert (
            general.unwrap("""
            Hello, world!
            This is a test.
            
            This is another test.
        """)
            == "Hello, world! This is a test.\n\nThis is another test."
        )
