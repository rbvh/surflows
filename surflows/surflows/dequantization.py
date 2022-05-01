import torch
import torch.nn as nn
import math
from surflows import utils
from surflows.distributions import BetaBox, CorrelatedUniformBox

from surflows.rqs_flow.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform, InverseTransform
from surflows.rqs_flow.transforms import RandomPermutation, CompositeTransform

class DequantizationBase(nn.Module):
    """
    Dequantization transform
    Encodes data that looks like continuous, discrete to [continuous, discrete_dequantized]
    Args:
        num_categories_per_dim : Tensor of shape [categorical_dims] that lists the number of categories per dimension
        continuous_size: Size of the continuous component of the data
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__()

        self.register_buffer("num_categories_per_dim_", num_categories_per_dim.long())

        self.num_continuous_dims_ = continuous_size
        self.num_discrete_dims_   = num_categories_per_dim.shape[0]

        # Precompute the rescaling log prob
        self.log_prob_rescaling_ = torch.log(self.num_categories_per_dim_.float()).sum()

        # Compute the factors required for the encoding of categorical labels 
        category_factors = torch.ones(self.num_discrete_dims_).long()
        for i in range(1, self.num_discrete_dims_):
            category_factors[i] = category_factors[i-1]*self.num_categories_per_dim_[i-1]
        self.register_buffer("category_factors_", category_factors)

        # The total number of categories
        self.num_categories_ = torch.prod(self.num_categories_per_dim_).item()

    # The number of flow dimensions
    def num_flow_dimensions(self):
        return self.num_continuous_dims_ + self.num_discrete_dims_

    # Sample the dequantizer
    def _sample_dequantizer(self, categories):
        raise NotImplementedError()

    # Logprob of the dequantizer
    def _log_prob_dequantizer(self, inputs, categories):
        raise NotImplementedError()

    def forward(self, inputs_continuous, inputs_categorical):
        outputs = torch.empty((inputs_categorical.shape[0], self.num_flow_dimensions()), dtype=torch.float32, device=inputs_categorical.device)

        # Add in the continuous data
        if inputs_continuous is not None:
            outputs[:,:self.num_continuous_dims_] = inputs_continuous

        # Encode categories into a single decimal
        if self.category_factors_.shape[0] != 1:
            categories_decimal = torch.sum(self.category_factors_*inputs_categorical, dim=-1)
        else:
            categories_decimal = inputs_categorical.squeeze()

        # Generate noise and compute log prob to dequantize the categorical dims
        dequantization_noise, dequantization_log_probs = self._sample_dequantizer(categories_decimal)

        # Add the noise and rescale
        outputs[:,self.num_continuous_dims_:] = (inputs_categorical + dequantization_noise)/self.num_categories_per_dim_

        # Fix any outputs that are almost 1
        outputs[outputs > 0.999999] = 0.999999

        return outputs, dequantization_log_probs - self.log_prob_rescaling_

    def inverse(self, inputs):
        # split up the inputs
        if self.num_continuous_dims_ == 0:
            outputs_continuous = None
            inputs_dequantized = inputs
        else:
            outputs_continuous = inputs[:,:self.num_continuous_dims_]
            inputs_dequantized = inputs[:,self.num_continuous_dims_:]

        # Fix any outputs that are almost 1
        inputs_dequantized[inputs_dequantized > 0.999999] = 0.999999

        # Scale with the number of categories
        inputs_dequantized = inputs_dequantized*self.num_categories_per_dim_

        # Round to get the quantized output
        outputs_discrete = torch.floor(inputs_dequantized).long()

        # Encode to decimal
        if self.category_factors_.shape[0] != 1:
            categories_decimal = torch.sum(self.category_factors_*outputs_discrete, dim=-1)
        else:
            categories_decimal = outputs_discrete.squeeze()

        # Mod to isolate the noise, and evaluate the dequantization log prob
        dequantization_log_prob = self._log_prob_dequantizer(inputs_dequantized % 1, categories_decimal)

        return outputs_continuous, outputs_discrete, -dequantization_log_prob + self.log_prob_rescaling_

class DequantizationUniform(DequantizationBase):
    """
    Implementation of dequantization layer with uniform dequantization
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        # Put an empty tensor in the state dict to infer the device from
        self.register_buffer("aux_", torch.empty([1]))

    def _sample_dequantizer(self, categories_decimal):
        return torch.rand(categories_decimal.shape[0], self.num_discrete_dims_, device=self.aux_.device), torch.zeros(categories_decimal.shape[0], device=self.aux_.device)

    def _log_prob_dequantizer(self, inputs, categories_decimal):
        return torch.zeros(categories_decimal.shape[0], device=categories_decimal.device)

class DequantizationBeta(DequantizationBase):
    """
    Implementation of dequantization layer with beta distribution dequantization
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        self.beta_ = BetaBox()

        self.concentration_ = nn.Parameter(torch.ones(self.num_categories_, num_categories_per_dim.shape[0], 2))

    def _sample_dequantizer(self, categories_decimal):
        concentrations = self.concentration_[categories_decimal]

        sample = self.beta_.sample(1, concentrations).squeeze(dim=0)
        return sample, utils.sum_except_batch(self.beta_.log_prob(sample, concentrations))
        
    def _log_prob_dequantizer(self, inputs, categories_decimal):
        concentrations = self.concentration_[categories_decimal]

        # Compute the log-probs
        return utils.sum_except_batch(self.beta_.log_prob(inputs, concentrations))

class DequantizationCorrelatedUniform(DequantizationBase):
    """
    Implementation of dequantization layer with correlated uniform distribution dequantization
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        self.correlated_uniform_ = CorrelatedUniformBox()

        # A set of Cholesky parameters for every category
        self.cholesky_ = nn.Parameter(torch.ones(self.num_categories_, int(continuous_size*(continuous_size-1)/2)))

    def _sample_dequantizer(self, categories):
        # Get the relevant Cholesky parameters
        choleskies = self.cholesky_[categories]

        sample = self.correlated_uniform_.sample(1, choleskies).squeeze(dim=0)
        return sample, utils.sum_except_batch(self.correlated_uniform_.log_prob(sample, choleskies))
    
    def _log_prob_dequantizer(self, inputs, encoding):
        # Get the relevant Cholesky parameters
        choleskies = self.cholesky_[encoding]

        return utils.sum_except_batch(self.correlated_uniform_.log_prob(inputs, choleskies))

class DequantizationFlow(DequantizationBase):
    """
    Implementation of dequantization layer with conditional autoregressive flow dequantization
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        # Hardcoded hyperparameters
        n_layers = 3
        n_knots = 8
        n_made_layers = 2
        n_made_units_per_dim = 2
        n_embedding = math.ceil(self.num_categories_**0.25)

        # Embedding
        self.embedding_ = nn.Embedding(self.num_categories_, n_embedding)

        # Flow transform - Use an inverse transform because we need the sampling step to be fast
        flow_transforms = []
        for _ in range(n_layers):
            flow_transforms.append(RandomPermutation(features=self.num_discrete_dims_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.num_discrete_dims_, 
                hidden_features=n_made_units_per_dim*self.num_discrete_dims_,
                context_features=n_embedding,
                num_bins=n_knots,
                num_blocks=n_made_layers,
                tails="restricted",
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

    def _sample_dequantizer(self, categories):
        # Unit box sample
        samples = torch.rand(categories.shape[0], self.num_discrete_dims_, device=categories.device)

        # Embed categories
        categories_embedded = self.embedding_(categories)

        # Transform and return
        return self.composite_flow_transform_.forward(samples, context=categories_embedded)

    def _log_prob_dequantizer(self, inputs, categories):
        # Embed categories
        categories_embedded = self.embedding_(categories)

        # Transform
        _, log_prob = self.composite_flow_transform_.inverse(inputs, context=categories_embedded)

        return -log_prob


