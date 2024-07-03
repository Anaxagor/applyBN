from rich.console import Console
from rich.traceback import install

# Initialize the console for rich
console = Console()
install(console=console)

class LibraryError(Exception):
    """
    Base class for all exceptions raised by the library.

    Parameters:
        message (str): Optional message for the exception.

    Usage Examples:
        >>> try:
        >>>     raise LibraryError("Custom library error message")
        >>> except LibraryError as e:
        >>>     print(f"Caught an exception: {e}")
    """
    def __init__(self, message=None):
        if message is None:
            message = "An error occurred in applybn."
        super().__init__(message)
        console.print_exception()

class InvalidInputError(LibraryError):
    """
    Exception raised for invalid input.

    Parameters:
        message (str): Optional message for the exception.

    Usage Examples:
        >>> try:
        >>>     raise InvalidInputError("Input must be an integer.")
        >>> except InvalidInputError as e:
        >>>     print(f"Caught an exception: {e}")
    """
    def __init__(self, message=None):
        if message is None:
            message = "Invalid input provided."
        super().__init__(message)
        console.print_exception()

class OperationFailedError(LibraryError):
    """
    Exception raised when an operation fails.

    Parameters:
        message (str): Optional message for the exception.

    Usage Examples:
        >>> try:
        >>>     raise OperationFailedError("Operation failed due to unknown error.")
        >>> except OperationFailedError as e:
        >>>     print(f"Caught an exception: {e}")
    """
    def __init__(self, message=None):
        if message is None:
            message = "The operation failed to complete successfully."
        super().__init__(message)
        console.print_exception()

class ResourceNotFoundError(LibraryError):
    """
    Exception raised when a required resource is not found.

    Parameters:
        message (str): Optional message for the exception.

    Usage Examples:
        >>> try:
        >>>     raise ResourceNotFoundError("Requested resource was not found.")
        >>> except ResourceNotFoundError as e:
        >>>     print(f"Caught an exception: {e}")
    """
    def __init__(self, message=None):
        if message is None:
            message = "The requested resource was not found."
        super().__init__(message)
        console.print_exception()
