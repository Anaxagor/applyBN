class LibraryError(Exception):
    """Base class for all exceptions raised by the library."""
    def __init__(self, message=None):
        if message is None:
            message = "An error occurred in applybn."
        super().__init__(message)


class InvalidInputError(LibraryError):
    """Exception raised for invalid input."""
    def __init__(self, message=None):
        if message is None:
            message = "Invalid input provided."
        super().__init__(message)


class OperationFailedError(LibraryError):
    """Exception raised when an operation fails."""
    def __init__(self, message=None):
        if message is None:
            message = "The operation failed to complete successfully."
        super().__init__(message)


class ResourceNotFoundError(LibraryError):
    """Exception raised when a required resource is not found."""
    def __init__(self, message=None):
        if message is None:
            message = "The requested resource was not found."
        super().__init__(message)
