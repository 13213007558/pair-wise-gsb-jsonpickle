"""
Stores custom jsonpickle errors.
"""

from typing import Any


class ClassNotFoundError(BaseException):
    def __init__(*args: Any, **kwargs: Any):
        pass


class ExternalReferenceError(Exception):
    """Base error for the py/external external reference protocol."""


class ExternalReferenceNotFoundError(ExternalReferenceError, KeyError):
    """The resolver reported that an external identifier does not exist."""


class ExternalResolverError(ExternalReferenceError):
    """The external resolver raised an unexpected error."""


class ExternalReferenceTypeError(ExternalReferenceError, TypeError):
    """A resolved external object does not match its declared type."""
