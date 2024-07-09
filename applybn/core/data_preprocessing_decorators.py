import functools


def copy_data(method):
    """
    Decorator to copy data before applying any transformation
    :param method:
    :return:
    """

    @functools.wraps(method)
    def wrapper(self, data, target):
        data = data.copy()
        return method(self, data, target)

    return wrapper
