import numpy as np
import torch
from torch.nn import functional as F

from surflows import utils
from surflows.distributions import Distribution


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(num_samples, context=embedded_context)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = utils.merge_leading_dims(noise, num_dims=2)
            embedded_context = utils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples    