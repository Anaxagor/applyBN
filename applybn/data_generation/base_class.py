from bamt.networks.base import BaseNetwork


class DataGenerator:
    """
    Base class for data generation methods

    Attributes:
        bn (BaseNetwork): Bayesian Network for synthetic data generation.
    """

    def __init__(
            self,
            bn: BaseNetwork
    ):
        self.bn = bn
