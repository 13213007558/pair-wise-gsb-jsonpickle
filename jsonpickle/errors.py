"""
Stores custom jsonpickle errors.
"""

from typing import Any


class ClassNotFoundError(BaseException):
    def __init__(*args: Any, **kwargs: Any):
        pass


class ExternalReferenceError(Exception):
    """Base class for errors raised by the external reference protocol."""


class ExternalReferenceNotFoundError(ExternalReferenceError):
    """Raised when a resolver cannot find the requested external identifier."""


class ExternalResolverError(ExternalReferenceError):
    """Raised when the external resolver itself fails or is misconfigured."""


class ExternalReferenceTypeError(ExternalReferenceError, TypeError):
    """Raised when a resolved object does not match the declared type."""
