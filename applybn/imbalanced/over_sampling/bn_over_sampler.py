from imblearn.over_sampling.base import BaseOverSampler


class BNOverSampler(BaseOverSampler):
    """Bayesian Network over-sampler."""

    # TODO: Implement Bayesian Network over-sampler using BaseOverSampler as reference
    # However, SamplerMixin, OneToOneFeatureMixin, BaseEstimator base classes from sklearn
    # may be more appropriate
    def __init__(self):
        super().__init__()

    def _fit_resample(self, X, y, **params):
        pass
