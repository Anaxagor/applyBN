import logging

from rich.logging import RichHandler


class Logger:
    """
    A logger class using rich and logging libraries.

    Attributes:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (str, optional): Path to a file to log messages. If None, logs will not be saved to a file.
        logger (logging.Logger): Configured logger instance.

    Methods:
        get_logger(): Returns the configured logger instance.

    Usage Examples:
        # Basic setup
        >>> logger = Logger("my_logger", level=logging.DEBUG)
        >>> log = logger.get_logger()
        >>> log.info("This is an info message")
        >>> log.debug("This is a debug message")

        # Setup with file logging
        >>> logger = Logger("file_logger", level=logging.INFO, log_file="my_log.log")
        >>> log = logger.get_logger()
        >>> log.info("This will be logged to the file")
    """

    def __init__(self, name, level=logging.INFO, log_file=None):
        """
        Initializes the Logger with a specified name, level, and optional log file.

        Parameters:
            name (str): Name of the logger.
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
            log_file (str, optional): Path to a file to log messages. If None, logs will not be saved to a file.
        """
        self.name = name
        self.level = level
        self.log_file = log_file
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        Sets up the logger with rich and logging libraries.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        # Create a console handler using RichHandler
        console_handler = RichHandler()
        console_handler.setLevel(self.level)

        # Create a formatter and set it for the console handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        if not logger.handlers:
            logger.addHandler(console_handler)

        # Optionally add a file handler if log_file is specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self):
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: Configured logger instance.
        """
        return self.logger
