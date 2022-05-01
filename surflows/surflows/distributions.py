"""Basic definitions for the distributions module."""

import torch
from torch import nn
from torch import distributions
from surflows import utils
from torch.distributions import categorical
import torch.nn.functional as F
import math as m
import numpy as np

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
class Distribution(nn.Module):
    """Base class for all distribution objects."""

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs, context=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        if context is not None:
            assert inputs.shape[0] == context.shape[0]

        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

class UniformBox(Distribution):
    def __init__(self, dim):
        super().__init__()

        self.dim_ = dim
        # Put an empty tensor in the state dict to infer the device from
        self.register_buffer("aux_", torch.empty([1]))
        
    def _log_prob(self, inputs, context=None):
        """
        Ignores the context
        """
        return torch.zeros(inputs.shape[0], device=inputs.device)

    def _sample(self, num_samples, context=None):
        return torch.rand(num_samples, self.dim_, device=self.aux_.device)

class BetaBox(Distribution):
    """
    Beta distribution, where the concentration variables are given by the context
    """
    def __init__(self):
        super().__init__()

    def _log_prob(self, inputs, context):
        if inputs.shape[0] != context.shape[0]:
            raise ValueError(
                "Input shape {} and context shape {} do not match".format(
                    inputs.shape[0], context[0].shape
                )
            )
        if context.shape[-1] != 2:
            raise ValueError(
                "Last context dim {} is not 2".format(
                    context.shape[-1]
                )
            )

        beta = distributions.Beta(context[...,0], context[...,1])
        return beta.log_prob(inputs)

    def _sample(self, num_samples, context):
        if context.shape[-1] != 2:
            raise ValueError(
                "Last context dim {} is not 2".format(
                    context.shape[-1]
                )
            )
        beta = distributions.Beta(context[...,0], context[...,1])
        return beta.sample(torch.Size([num_samples]))

class CorrelatedUniformBox(Distribution):
    """
    Correlated beta distribution, accomplished through a copula starting from a 
    multivariate Gaussian N(0, Sigma) with covariance matrix with unit variances

    The context is the lower-triangular Cholesky decomposition of the covariance matrix
    inputs:  tensor with shape [batch_size, dim]
    context: tensor with shape [batch_size, 0.5*dim*(dim-1)] of unconstrained parameters
    """

    def __init__(self):
        super().__init__()

    def _transform_context_to_covariance(self, context):
        dim = int(round(int(0.5*(1 + m.sqrt(1 + 8*context.shape[-1])))))

        theta = m.pi*torch.exp(context)/(torch.exp(context) + 1)

        # Construct Cholesky decomposition in the spherical representation
        L_shape = list(context.shape[:-1])
        L_shape.append(dim)
        L_shape.append(dim)
        L = torch.zeros(L_shape, device=context.device)

        tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1, device=context.device)
        L[...,torch.arange(dim), torch.arange(dim)] = 1
        L[...,tril_indices[0], tril_indices[1]] = torch.cos(theta)
        L_clone = L.clone()

        for i in range(dim-1):
            L[...,:,i+1:] *= torch.sqrt(1 - L_clone[...,:,i]**2).unsqueeze(-1)

        # Add a small fraction of the identity matrix to avoid numerical 
        # issues with positive definiteness of the covariance
        cov = torch.matmul(L, L.transpose(-2,-1)) + 0.00001*torch.eye(dim, device=context.device)

        return cov

    def _log_prob(self, inputs, context):
        if inputs.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                "Input shape {} and context shape {} do not match".format(
                    inputs.shape[0], context[0].shape
                )
            )

        # Transform to normal distribution
        inputs_transform = m.sqrt(2.)*torch.erfinv((2.*inputs-1.).clamp(-1+1e-13,1-1e-13))

        # Evaluate the covariance matrix from the context
        cov = self._transform_context_to_covariance(context)

        # Set up the distribution and return log prob
        multinomial = torch.distributions.MultivariateNormal(torch.zeros_like(inputs), covariance_matrix=cov)

        log_probs_correlated = multinomial.log_prob(inputs_transform)
        log_probs_uncorrelated = torch.sum(-inputs_transform**2/2 - m.log(2*m.pi)/2, dim=-1)

        return log_probs_correlated - log_probs_uncorrelated

    def _sample(self, num_samples, context):
        # Evaluate the covariance matrix from the context
        cov = self._transform_context_to_covariance(context)
        dim = cov.shape[-1]

        # Sample from base multinomial
        multinomial = torch.distributions.MultivariateNormal(torch.zeros((context.shape[0], dim), device=context.device), covariance_matrix=cov)
        sample = multinomial.sample(torch.Size([num_samples]))

        # Transform to uniform
        return 0.5*(1.+torch.erf(sample/m.sqrt(2.)))

class StandardNormal(Distribution):
    """
    Simple standard normal distribution
    """
    def __init__(self, shape):
        super().__init__()
        self.shape_ = shape

        self.register_buffer("log_z_", torch.tensor(0.5 * shape * np.log(2 * np.pi), dtype=torch.float64))

    def _log_prob(self, inputs, context):
        return -0.5*torch.sum(inputs**2, dim=1) - self.log_z_
    
    def _sample(self, num_samples, context):
        return torch.randn(num_samples, self.shape_, device=self.log_z_.device)

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------- Mixture distribution--------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
class MixtureDistribution(nn.Module):
    """
    Base class for a mixed categorical-continuous distribution
    Models the joint likelihood p(x,r) of a discrete variable r and a 
    continuous variable x as p(x,r) = p(r) * p(x|r).
    The base class implements a tabulated model for p(r), while p(x|r) remains to be 
    implemented in a derived class.
    """

    def __init__(self, dim_continuous, num_categories):
        super().__init__()

        self.dim_continuous_ = dim_continuous
        self.num_categories_ = num_categories

        self.register_buffer("categorical_log_probs_", torch.zeros(num_categories))

    def add_to_counts(self, data, weights=None):
        '''
        Data should be a (batch_size)-size list of categorical indices
        '''        
        if weights is not None:
            self.categorical_counts_.index_add_(0, data, weights.float())
        else:
            self.categorical_counts_.index_add_(0, data, torch.ones(data.shape[0], device=data.device))
    
    def compute_categorical_log_probs(self):
        sum_of_counts = torch.sum(self.categorical_counts_)
        self.categorical_log_probs_ = torch.log(self.categorical_counts_/sum_of_counts)

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob_conditional(self, inputs_continuous, inputs_categorical):
        assert inputs_continuous.shape[0] == inputs_categorical.shape[0]

        return self._log_prob_conditional(inputs_continuous, inputs_categorical)

    def log_prob_joint(self, inputs_continuous, inputs_categorical):
        assert inputs_continuous.shape[0] == inputs_categorical.shape[0]

        return self._log_prob_conditional(inputs_continuous, inputs_categorical) + self.categorical_log_probs_[inputs_categorical]

    def sample(self, num_samples, batch_size=None):
        assert num_samples > 0

        if batch_size is None:
            return self._sample(num_samples)

        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples_continuous = []
            samples_discrete = []
            for _ in range(num_batches):
                sample_continuous, sample_discrete = self._sample(batch_size)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)

            if num_leftover > 0:
                sample_continuous, sample_discrete = self._sample(num_leftover)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)

            return torch.cat(samples_continuous, dim=0), torch.cat(samples_discrete, dim=0)
            
    def sample_conditional(self, inputs_categorical, batch_size=None):
        if batch_size is None:
            return self._sample_conditional(inputs_categorical)

        else:
            num_batches = inputs_categorical.shape[0] // batch_size
            num_leftover = inputs_categorical.shape[0] % batch_size
            samples_continuous = []
            for i in range(num_batches):
                samples_continuous.append(self._sample_conditional(inputs_categorical[i*batch_size : (i+1)*batch_size]))

            if num_leftover > 0:
                samples_continuous.append(self._sample_conditional(inputs_categorical[num_batches*batch_size : ]))

        return torch.cat(samples_continuous, dim=0)

    def _sample(self, num_samples):
        assert ~torch.all(self.categorical_log_probs_ == 0.)

        # Gumbel-max trick to sample categorical distribution
        categorical_noise = torch.rand(num_samples, self.num_categories_, device=self.categorical_log_probs_.device)
        categorical_samples = torch.argmax(self.categorical_log_probs_ - torch.log(-torch.log(categorical_noise)), dim=-1)

        return self._sample_conditional(categorical_samples), categorical_samples


    def _sample_conditional(self, inputs_categorical):
        raise NotImplementedError()

    def _log_prob_conditional(self, inputs_continuous, inputs_categorical):
        raise NotImplementedError()

class UniformMixtureBox(MixtureDistribution):
    def __init__(self, dim_continuous, num_categories):
        super().__init__(dim_continuous, num_categories)
    
    def _sample_conditional(self, inputs_categorical):
        return torch.rand(inputs_categorical.shape[0], self.dim_continuous_, device=inputs_categorical.device)

    def _log_prob_conditional(self, inputs_continuous, inputs_categorical):
        assert inputs_continuous.shape[0] == inputs_categorical.shape[0]

        return torch.zeros(inputs_categorical.shape[0], device=inputs_continuous.device)

class BetaMixtureBox(MixtureDistribution):
    def __init__(self, dim_continuous, num_categories):
        super().__init__(dim_continuous, num_categories)

        self.beta_ = BetaBox()

        # Concentration parameters of the Beta distribution
        self.concentration_ = nn.Parameter(torch.ones(self.num_categories_, self.dim_continuous_, 2))

    def _sample_conditional(self, inputs_categorical):
        concentrations = self.concentration_[inputs_categorical]

        return self.beta_.sample(1, concentrations).squeeze(dim=0)

    def _log_prob_conditional(self, inputs_continuous, inputs_categorical):
        # Get the relevant log-concentrations
        concentrations = self.concentration_[inputs_categorical]

        # Protection against infinities due to numbers on the edge
        inputs_continuous[inputs_continuous == 1.] -= 1e-6
        inputs_continuous[inputs_continuous == 0.] += 1e-6

        log_probs = torch.zeros_like(inputs_continuous)
        mask = inputs_continuous >= 0.
        log_probs[mask] = self.beta_.log_prob(inputs_continuous[mask], concentrations[mask])
        
        # Sum over dimensions and return
        return utils.sum_except_batch(log_probs)

class StandardMixtureNormal(MixtureDistribution):
    def __init__(self, dim_continuous, num_categories):
        super().__init__(dim_continuous, num_categories)

    def _sample_conditional(self, inputs_categorical):
        return torch.randn(inputs_categorical.shape[0], self.dim_continuous_, device=inputs_categorical.device)

    # NOTE: This one can get called with nans, which need to be filtered out
    def _log_prob_conditional(self, inputs_continuous, inputs_categorical):
        log_prob_conditional = torch.zeros_like(inputs_continuous)
        log_prob_conditional[~inputs_continuous.isnan()] = -0.5*inputs_continuous[~inputs_continuous.isnan()]**2 - 0.5*np.log(2*np.pi)

        return log_prob_conditional.sum(dim=1)
