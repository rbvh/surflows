import torch
import torch.nn as nn
import math
from ppflows import utils
from ppflows.distributions import BetaBox, CorrelatedUniformBox

from ppflows.rqs_flow.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform, InverseTransform
from ppflows.rqs_flow.transforms import RandomPermutation, CompositeTransform

class ArgmaxBase(nn.Module):
    """
    Argmax transform with a binary encoding
    Encodes data that looks like continuous, categorical to [continuous, categorical_dequantized]
    Args:
        num_categories_per_dim : Tensor of shape [categorical_dims] that lists the number of categories per dimension
        continuous_size: Size of the continuous component of the data
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__()

        self.register_buffer("num_categories_per_dim_", num_categories_per_dim.long())
        self.num_continuous_dims_ = continuous_size

        # Compute the factors required for the encoding of categorical labels 
        category_factors = torch.ones(self.num_categories_per_dim_.shape[0]).long()
        for i in range(1, self.num_categories_per_dim_.shape[0]):
            category_factors[i] = category_factors[i-1]*self.num_categories_per_dim_[i-1]
        self.register_buffer("category_factors_", category_factors)

        # The total number of categories
        self.num_categories_ = torch.prod(self.num_categories_per_dim_).item()
        
        # The number of required dims in binary
        self.num_binary_dims_ = max(1, math.ceil(math.log2(self.num_categories_)))

        # Mask required for encoding into binary
        self.register_buffer("binary_mask_", 2 ** torch.arange(self.num_binary_dims_ - 1, -1, -1))

    # The number of flow dimensions
    def num_flow_dimensions(self):
        return 2*self.num_binary_dims_ + self.num_continuous_dims_

    # Sample the dequantizer
    def _sample_dequantizer(self, categories):
        raise NotImplementedError()

    # Logprob of the dequantizer
    def _log_prob_dequantizer(self, inputs, categories):
        raise NotImplementedError()

    def forward(self, inputs_continuous, inputs_categorical):
        batch_size = inputs_categorical.shape[0]

        # Encode the categorical data into single decimal form
        # Skip encoding if only one categorical dimension
        if self.category_factors_.shape[0] != 1:
            decimal_encoded = torch.sum(self.category_factors_*inputs_categorical, dim=-1)
        else:
            decimal_encoded = inputs_categorical.squeeze(-1)

        # Next, encode to binary
        binary_encoded = decimal_encoded.unsqueeze(-1).bitwise_and(self.binary_mask_).ne(0)
        
        # Generate noise and compute log prob to dequantize the categorical dims
        dequantization_noise, dequantization_log_probs = self._sample_dequantizer(decimal_encoded)

        # Compute index tensors that select the relevant elements of the noise
        indices_max = torch.arange(0,2*self.num_binary_dims_, 2, device=inputs_categorical.device).repeat(batch_size,1)
        indices_min = torch.arange(0,2*self.num_binary_dims_, 2, device=inputs_categorical.device).repeat(batch_size,1)
        indices_max[~binary_encoded] += 1
        indices_min[binary_encoded] += 1

        # Rescale the noise to ensure argmax returns the correct result
        # We need to jump through some hoops to get array indexing to work here
        # Flatten the noise tensor and the index tensors
        offset = 2*self.num_binary_dims_*torch.arange(0, batch_size, device=inputs_categorical.device)[:,None]
        indices_max_flat = (indices_max + offset).flatten()
        indices_min_flat = (indices_min + offset).flatten()

        # Record the logprob of the upcoming multiplication
        rescaling_log_probs = -torch.log(dequantization_noise.flatten()[indices_max_flat]).reshape(batch_size, self.num_binary_dims_).sum(dim=-1)

        # Multiply to adhere to argmax
        dequantization_noise.flatten()[indices_min_flat] *= dequantization_noise.flatten()[indices_max_flat]

        # Concat the new stuff to the old
        inputs_continuous = torch.cat((inputs_continuous, dequantization_noise), dim=1)

        return inputs_continuous, dequantization_log_probs - rescaling_log_probs

    def inverse(self, inputs):
        batch_size = inputs.shape[0]

        if self.num_continuous_dims_ == 0:
            outputs_continuous = None
            inputs_dequantized = inputs
        else:
            outputs_continuous = inputs[:,:self.num_continuous_dims_]
            inputs_dequantized = inputs[:,self.num_continuous_dims_:]

        # Set up a mask to select the minima/maxima in the binary dimension
        inputs_dequantized_view = inputs_dequantized.view(batch_size, self.num_binary_dims_, 2)
        max_mask = torch.zeros_like(inputs_dequantized_view).bool()
        max_mask[...,0] = inputs_dequantized_view[...,0] >= inputs_dequantized_view[...,1]
        max_mask[...,1] = inputs_dequantized_view[...,1] > inputs_dequantized_view[...,0]
        min_mask = ~max_mask

        # Get the binary encoding
        binary_encoded = ~torch.argmax(max_mask.long(), dim=-1).bool()
        # Encode from binary to decimal 
        decimal_encoded = torch.sum(self.binary_mask_ * binary_encoded, -1)

        # Compute the scaling log prob
        rescaling_log_probs = -torch.log(inputs_dequantized_view[max_mask].view(batch_size, self.num_binary_dims_)).sum(dim=-1)

        # Invert the scaling
        inputs_dequantized_view_copy = inputs_dequantized_view.clone()
        inputs_dequantized_view_copy[min_mask] /= inputs_dequantized_view_copy[max_mask]

        # Compute the dequantization log prob
        dequantization_log_probs = self._log_prob_dequantizer(inputs_dequantized_view_copy.flatten(start_dim=1), decimal_encoded)

        # Finally revert to original encoding
        outputs_categorical = torch.empty((batch_size, self.num_categories_per_dim_.shape[0]), dtype=torch.long, device=inputs.device)
        for i in range(self.num_categories_per_dim_.shape[0]-1, -1, -1):
            outputs_categorical[:,i] = (decimal_encoded - decimal_encoded % self.category_factors_[i])/self.category_factors_[i]
            decimal_encoded -= outputs_categorical[:,i]*self.category_factors_[i]

        return outputs_continuous, outputs_categorical, -dequantization_log_probs + rescaling_log_probs


class ArgmaxUniform(ArgmaxBase):
    """
    Implementation of argmax layer with uniform dequantization
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        # Put an empty tensor in the state dict to infer the device from
        self.register_buffer("aux_", torch.empty([1]))

    def _sample_dequantizer(self, encoding):
        assert(~torch.any(encoding >= self.num_categories_))

        return utils.exclusive_rand(encoding.shape[0], 2*self.num_binary_dims_, device=encoding.device), torch.zeros(encoding.shape[0], device=encoding.device)

    def _log_prob_dequantizer(self, inputs, encoding):
        return torch.zeros(encoding.shape[0], device=self.aux_.device)

class ArgmaxBeta(ArgmaxBase):
    """
    Implementation of argmax layer with beta distribution conditional likelihood
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        self.beta_ = BetaBox()

        # A set of alphas and betas for every category
        self.concentration_ = nn.Parameter(torch.ones(self.num_categories_, 2*self.num_binary_dims_, 2))
    
    def _sample_dequantizer(self, encoding):
        assert(~torch.any(encoding >= self.num_categories_))

        # Get the relevant concentrations
        concentrations = self.concentration_[encoding]

        sample = self.beta_.sample(1, concentrations).squeeze(dim=0)

        return sample, utils.sum_except_batch(self.beta_.log_prob(sample, concentrations))

    def _log_prob_dequantizer(self, inputs, encoding):
        # A mask to discern samples with a categorical index > self.num_categories_
        mask_category_exists = encoding < self.num_categories_

        # Get the relevant log-concentrations
        concentrations = self.concentration_[encoding[mask_category_exists]]

        log_probs = torch.ones(inputs.shape[0], 2*self.num_binary_dims_)*(-float('inf'))
        log_probs[mask_category_exists] = self.beta_.log_prob(inputs[mask_category_exists], concentrations[mask_category_exists])

        return utils.sum_except_batch(log_probs)

class ArgmaxCorrelatedUniform(ArgmaxBase):
    """
    Implementation of argmax layer with correlated uniform conditional likelihood
    """
    def __init__(self, num_categories_per_dim, continuous_size):
        super().__init__(num_categories_per_dim, continuous_size)

        self.correlated_uniform_ = CorrelatedUniformBox()

        # A set of Cholesky parameters for every category
        self.cholesky_ = nn.Parameter(torch.ones(self.num_categories_, int(continuous_size*(continuous_size-1)/2)))

    def _sample_dequantizer(self, categories):
        assert(~torch.any(encoding >= self.num_categories_))
        
        # Get the relevant Cholesky parameters
        choleskies = self.cholesky_[categories]

        sample = self.correlated_uniform_.sample(1, choleskies).squeeze(dim=0)

        return sample, utils.sum_except_batch(self.correlated_uniform_.log_prob(sample, choleskies))

    def _log_prob_dequantizer(self, inputs, encoding):
        # A mask to discern samples with a categorical index > self.num_categories_
        mask_category_exists = encoding < self.num_categories_

        # Get the relevant Cholesky parameters
        choleskies = self.cholesky_[encoding[mask_category_exists]]

        log_probs = torch.ones(inputs.shape[0], 2*self.num_binary_dims_)*(-float('inf'))
        log_probs[mask_category_exists] = self.correlated_uniform_.log_prob(inputs[mask_category_exists], choleskies[mask_category_exists])

        return utils.sum_except_batch(log_probs)

class ArgmaxFlow(ArgmaxBase):
    """
    Implementation of argmax layer with conditional autoregressive flow dequantization
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
            flow_transforms.append(RandomPermutation(features=2*self.num_binary_dims_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=2*self.num_binary_dims_, 
                hidden_features=n_made_units_per_dim*2*self.num_binary_dims_,
                context_features=n_embedding,
                num_bins=n_knots,
                num_blocks=n_made_layers,
                tails="restricted",
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

    def _sample_dequantizer(self, encoding):
        assert(~torch.any(encoding >= self.num_categories_))

        # Unit box sample
        samples = utils.exclusive_rand(encoding.shape[0], 2*self.num_binary_dims_, device=encoding.device)

        # Embed encoding
        encoding_embedded = self.embedding_(encoding)

        # Transform and return
        return self.composite_flow_transform_(samples, context=encoding_embedded)
    
    def _log_prob_dequantizer(self, inputs, encoding):
        # A mask to discern samples with a categorical index > self.num_categories_
        mask_category_exists = encoding < self.num_categories_

        # Embed categories
        categories_embedded = self.embedding_(encoding[mask_category_exists])

        # Transform
        _, log_probs_exist = self.composite_flow_transform_.inverse(inputs[mask_category_exists], context=categories_embedded)

        log_probs = torch.ones(inputs.shape[0])*(-float('inf'))
        log_probs[mask_category_exists] = log_probs_exist

        return -log_probs