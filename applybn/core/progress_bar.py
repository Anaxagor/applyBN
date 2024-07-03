from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressBar:
    """
    A class to use Rich progress bar across all modules.

    Methods:
        create_task(description, total): Creates a progress task with a description and total steps.
        update_task(task_id, advance): Advances the progress of a task.
        start(): Starts the progress bar.
        stop(): Stops the progress bar.

    Usage Examples:
        >>> pb = ProgressBar()
        >>> task_id = pb.create_task("Processing", total=100)
        >>> pb.start()
        >>> for i in range(100):
        >>>     pb.update_task(task_id, advance=1)
        >>> pb.stop()
    """

    def __init__(self):
        """
        Initializes the ProgressBar with a rich progress instance.
        """
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.task_id = None

    def create_task(self, description, total):
        """
        Creates a progress task with a description and total steps.

        Parameters:
            description (str): Description of the task.
            total (int): Total steps for the task.

        Returns:
            int: Task ID of the created task.
        """
        self.task_id = self.progress.add_task(description, total=total)
        return self.task_id

    def update_task(self, task_id, advance):
        """
        Advances the progress of a task.

        Parameters:
            task_id (int): ID of the task to update.
            advance (int): Number of steps to advance.
        """
        self.progress.update(task_id, advance=advance)

    def start(self):
        """
        Starts the progress bar.
        """
        self.progress.start()

    def stop(self):
        """
        Stops the progress bar.
        """
        self.progress.stop()
