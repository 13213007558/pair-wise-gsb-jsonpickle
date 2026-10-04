"""
Stores custom jsonpickle errors.
"""

from typing import Any


class ClassNotFoundError(BaseException):
    def __init__(*args: Any, **kwargs: Any):
        pass


class InspectError(ValueError):
    """Base class for errors found while inspecting a jsonpickle snapshot."""


class MalformedSnapshotError(InspectError):
    """Raised when a snapshot does not have the required wire shape."""


class InvalidReferenceError(MalformedSnapshotError):
    """Raised when a reference does not point to a valid snapshot object."""


class DuplicateIdentityError(MalformedSnapshotError):
    """Raised when one structure claims more than one snapshot identity."""


class LegacyFormatError(MalformedSnapshotError):
    """Raised when an old, non-describable snapshot format is encountered."""
