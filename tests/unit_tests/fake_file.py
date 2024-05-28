from io import BytesIO

from app.core.parsing import File


class FakeFile(File):
    """A fake file for testing purposes"""

    @classmethod
    def from_bytes(cls, file: BytesIO) -> "FakeFile":
        return NotImplemented
