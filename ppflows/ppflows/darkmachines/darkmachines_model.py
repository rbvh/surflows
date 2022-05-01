import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import yaml

from ppflows.rqs_flow.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from ppflows.rqs_flow.transforms import RandomPermutation, CompositeTransform

from ppflows.darkmachines.darkmachines_auxiliary import DarkMachinesData

from ppflows.distributions import StandardMixtureNormal
from ppflows.permuters import StochasticPermutation, IteratedPermutation
from ppflows.rqs_flow.transforms import Permutation

from ppflows.utils import exclusive_rand

'''
Model that includes permutation, dropout and discrete mixture for darkmachines data
'''

class DarkMachinesMixtureModel(nn.Module):
    def __init__(self, hyperparams):
        super(DarkMachinesMixtureModel, self).__init__()

        # Number of continuous dimensions
        self.flow_dim_ = 2 + 4*hyperparams["num_objects"]

        # Number of categories per dim
        self.num_categories_per_dim_ = torch.ones(hyperparams["num_objects"])*8
        total_num_categories = int(torch.prod(self.num_categories_per_dim_).item())
    
        # Set up the permutation model
        permute_classes_continuous = []
        permute_classes_discrete = []
        for i in range(hyperparams["num_objects"]):
            permute_classes_continuous.append(np.array([2+4*i, 3+4*i, 4+4*i, 5+4*i]))
            permute_classes_discrete.append(np.array([i]))
        permute_classes_continuous = np.array(permute_classes_continuous)
        permute_classes_discrete = np.array(permute_classes_discrete)

        self.permuter_ = None
        if hyperparams["permutation"] == 'stochastic':
            self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)
            
        # ------------------------------------------------------------------------
        # Categorical encoding proceeds as follows:
        # - Particle IDs are converted into a category using category_factors_
        # - The category is converted into an index using category_to_index_
        # - The index is transformed to an embedding using embedding_
        # The inverse proceeds as:
        # - The index is converted into a category using index_to_category_
        # - The category is converted to particle IDs using category_factors_
        # ------------------------------------------------------------------------

        # Compute the factors required for the encoding of categorical labels 
        category_factors = torch.ones(self.num_categories_per_dim_.shape[0]).long()
        for i in range(1, self.num_categories_per_dim_.shape[0]):
            category_factors[i] = category_factors[i-1]*self.num_categories_per_dim_[i-1]
        self.register_buffer("category_factors_", category_factors, persistent=False)

        # -------------------- Load the categorical probs dict -------------------        
        with open(hyperparams["data_dir"] + "/dicts/category_dict_chan_" + hyperparams['channel'] + "_" + hyperparams['permutation'] + "_num_objects_" + str(hyperparams["num_objects"]) + ".yaml") as category_dict_data:
            category_dict = yaml.load(category_dict_data, Loader=yaml.Loader)

        category_to_index = torch.ones(total_num_categories).long()*(-1)
        index_to_category = torch.zeros(len(category_dict)).long()
        log_probs_categorical = torch.zeros(len(category_dict))

        # Fill in the maps between index and category
        for index, category in enumerate(category_dict):
            if category != "tot_samples":
                category_to_index[category] = index
                index_to_category[index] = category
                log_probs_categorical[index] = category_dict[category]

        self.register_buffer("category_to_index_", category_to_index, persistent=False)
        self.register_buffer("index_to_category_", index_to_category, persistent=False)
        total_samples = category_dict["tot_samples"]

        log_probs_categorical = torch.log(log_probs_categorical/total_samples)
        
        # Set the embedding
        self.embedding_ = nn.Embedding(len(category_dict), hyperparams["embedding_size"])

        # Base distribution
        self.base_dist_ = StandardMixtureNormal(self.flow_dim_, len(category_dict))
        self.base_dist_.categorical_log_probs_ = log_probs_categorical

        # Set up the flow model
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                context_features=hyperparams["embedding_size"],
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="linear", 
                tail_bound = 5
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

        # Set up a transform that only contains the permutations of the flow
        # Used for propagating the masking operation in the inverse direction
        permute_transforms = []
        for transform in flow_transforms:
            if isinstance(transform, RandomPermutation):
                permute_transforms.append(transform)
        self.composite_permute_transform_ = CompositeTransform(permute_transforms)

    # ----- Forward and backward passes -----
    def _forward(self, inputs_continuous, inputs_discrete):
        # Permute
        if self.permuter_ is not None:
            inputs_continuous, inputs_discrete, _ = self.permuter_.forward(inputs_continuous, inputs_discrete)

        # Encode the categorical data into single category
        outputs_discrete = torch.sum(self.category_factors_*inputs_discrete, dim=-1).squeeze()
        # Convert category to index
        outputs_discrete = self.category_to_index_[outputs_discrete]

        # Identify any data points with category not in the training set
        unknown_category_mask = outputs_discrete == -1
        
        # For the purposes of passing through the flow, set the category indices of unknown categories to 0
        outputs_discrete[unknown_category_mask] = 0
        
        # Pass through the flow network
        outputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous, context=self.embedding_(outputs_discrete.clone()))

        # Now correct for unknown categories
        outputs_discrete[unknown_category_mask] = -1
        transform_log_prob[unknown_category_mask] = -float("inf")
        
        return outputs_continuous, outputs_discrete, transform_log_prob

    def _inverse(self, inputs_continuous, inputs_discrete):
        # inputs_discrete is an index here - convert to category first
        inputs_discrete_index = self.index_to_category_[inputs_discrete]
        # Then convert to particle IDs
        outputs_discrete = torch.empty((inputs_continuous.shape[0], self.num_categories_per_dim_.shape[0]), dtype=torch.long, device=inputs_continuous.device)
        for i in range(self.num_categories_per_dim_.shape[0]-1, -1, -1):
            outputs_discrete[:,i] = (inputs_discrete_index - inputs_discrete_index % self.category_factors_[i])/self.category_factors_[i]
            inputs_discrete_index -= outputs_discrete[:,i]*self.category_factors_[i]

        # Perform dropout
        dropout_mask = torch.zeros_like(inputs_continuous).bool()
        # Empty entries
        dropout_mask[:,2:] = torch.repeat_interleave(outputs_discrete == 0, 4, dim=1)
        # Non-jet entries
        for i in range(outputs_discrete.shape[1]):
            dropout_mask[:,5+4*i] = torch.logical_or(outputs_discrete[:,i] == 0, outputs_discrete[:,i] > 2)
        
        # Pass the mask through the permute transform
        dropout_mask, _ = self.composite_permute_transform_.forward(dropout_mask)

        # Then apply dropout
        inputs_continuous[dropout_mask] = float('nan')

        # Pass through the flow network
        outputs_continuous, transform_log_prob = self.composite_flow_transform_.inverse(inputs_continuous, context=self.embedding_(inputs_discrete))
        
        # Permute
        if self.permuter_ is not None:
            outputs_continuous, outputs_discrete, _ = self.permuter_.inverse(outputs_continuous, outputs_discrete)

        return outputs_continuous, outputs_discrete, transform_log_prob

    def log_prob_conditional(self, inputs_continuous, inputs_discrete):
        inputs_continuous, inputs_discrete, total_log_prob = self._forward(inputs_continuous, inputs_discrete)

        # Evaluate the base distribution
        total_log_prob += self.base_dist_.log_prob_conditional(inputs_continuous, inputs_discrete)

        return total_log_prob

    def log_prob(self, inputs_continuous, inputs_discrete):
        inputs_continuous, inputs_discrete, total_log_prob = self._forward(inputs_continuous, inputs_discrete)

        # Evaluate the base distribution
        total_log_prob += self.base_dist_.log_prob_joint(inputs_continuous, inputs_discrete)

        return total_log_prob

    def sample(self, num_samples, batch_size=None):
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


    def sample_conditional(self, inputs_discrete, batch_size=None):
        if batch_size is None:
            return self._sample_conditional(inputs_discrete)

        else:
            num_batches = inputs_discrete.shape[0] // batch_size
            num_leftover = inputs_discrete.shape[0] % batch_size
            samples_continuous = []
            samples_discrete = []

            for i in range(num_batches):
                sample_continuous, sample_discrete = self._sample_conditional(inputs_discrete[i*batch_size : (i+1)*batch_size])
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
            
            if num_leftover > 0:
                sample_continuous, sample_discrete = self._sample_conditional(inputs_discrete[num_batches*batch_size : ])
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
        
        return torch.cat(samples_continuous, dim=0), torch.cat(samples_discrete, dim=0)


    # NOTE: This function does not unnormalize the data
    def _sample(self, num_samples):
        samples_continuous, samples_discrete = self.base_dist_.sample(num_samples)

        samples_continuous, samples_discrete, _ = self._inverse(samples_continuous, samples_discrete)
        return samples_continuous, samples_discrete

    # NOTE: This function expects particle ID categories
    # NOTE: For categories that are not in the training data, sampling is skipped
    def _sample_conditional(self, inputs_discrete):
        # Encode the categorical data into single category
        # Skip encoding if only one categorical dimension
        samples_discrete = torch.sum(self.category_factors_*inputs_discrete, dim=-1).squeeze()
        # Convert category to index
        samples_discrete = self.category_to_index_[samples_discrete]

        # Kick out -1's
        samples_discrete = samples_discrete[samples_discrete != -1]
    
        samples_continuous = self.base_dist_.sample_conditional(samples_discrete)

        samples_continuous, samples_discrete, _ = self._inverse(samples_continuous, samples_discrete)
        return samples_continuous, samples_discrete


class DarkMachinesDequantizationModel(nn.Module):
    def __init__(self, hyperparams):
        super(DarkMachinesDequantizationModel, self).__init__()

        # Number of continuous dimensions
        self.flow_dim_ = 2 + 5*hyperparams["num_objects"]

        # Set up the permutation model
        permute_classes_continuous = []
        permute_classes_discrete = []
        for i in range(hyperparams["num_objects"]):
            permute_classes_continuous.append(np.array([2+4*i, 3+4*i, 4+4*i, 5+4*i]))
            permute_classes_discrete.append(np.array([i]))
        permute_classes_continuous = np.array(permute_classes_continuous)
        permute_classes_discrete = np.array(permute_classes_discrete)

        self.permuter_ = None
        if hyperparams["permutation"] == 'stochastic':
            self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)

        # Compute the factors required for the encoding of dropout labels     
        dropout_factors = torch.ones(hyperparams["num_objects"]).long()
        for i in range(1, hyperparams["num_objects"]):
            dropout_factors[i] = dropout_factors[i-1]*2
        self.register_buffer("dropout_factors_", dropout_factors)

        # -------------------- Load the dropout probs dict -------------------        
        with open(hyperparams["data_dir"] + "/dicts/dropout_dict_chan_" + hyperparams['channel'] + "_" + hyperparams['permutation'] + "_num_objects_" + str(hyperparams["num_objects"]) + ".yaml") as dropout_dict_data:
            dropout_dict = yaml.load(dropout_dict_data, Loader=yaml.Loader)

        log_probs_dropout = torch.zeros(2**hyperparams["num_objects"])

        # Fill in the maps between index and category
        for dropout in dropout_dict:
            if dropout != "tot_samples":
                log_probs_dropout[dropout] = dropout_dict[dropout]
    
        total_samples = dropout_dict["tot_samples"]

        log_probs_dropout = torch.log(log_probs_dropout/total_samples)
        
        # Set the embedding
        self.embedding_ = nn.Embedding(2**hyperparams["num_objects"], hyperparams["embedding_size"])

        # Base distribution
        self.base_dist_ = StandardMixtureNormal(self.flow_dim_, 2**hyperparams["num_objects"])
        self.base_dist_.categorical_log_probs_ = log_probs_dropout

        # Set up the flow model
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                context_features=hyperparams["embedding_size"],
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="linear", 
                tail_bound = 5
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

        # Set up a transform that only contains the permutations of the flow
        # Used for propagating the masking operation in the inverse direction
        permute_transforms = []
        for transform in flow_transforms:
            if isinstance(transform, RandomPermutation):
                permute_transforms.append(transform)
        self.composite_permute_transform_ = CompositeTransform(permute_transforms)

    # ----- Forward and backward passes -----
    def log_prob(self, inputs_continuous, inputs_discrete):
        # Permute
        if self.permuter_ is not None:
            inputs_continuous, inputs_discrete, _ = self.permuter_.forward(inputs_continuous, inputs_discrete)

        # Find dropped objects
        dropped_objects = inputs_discrete == 0

        # Get the dropout label
        dropout_encoded = torch.sum(self.dropout_factors_*(inputs_discrete != 0), dim=-1).squeeze()

        # Dequantize and center around 0
        dequantized_discrete = inputs_discrete - 4.5 + exclusive_rand(inputs_discrete.shape[0], inputs_discrete.shape[1], device=inputs_continuous.device)

        # Drop from dequantized_discrete
        dequantized_discrete[dropped_objects] *= float('nan')

        # Cat continuous and discrete
        inputs_continuous = torch.cat((inputs_continuous, dequantized_discrete), dim=1)

        # Pass throught the flow
        inputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous, context=self.embedding_(dropout_encoded.clone()))

        # Evaluate the base distribution
        base_log_prob = self.base_dist_.log_prob_joint(inputs_continuous, dropout_encoded)

        return transform_log_prob + base_log_prob

class DarkMachinesClassifierModel(nn.Module):
    def __init__(self, hyperparams):
        super(DarkMachinesClassifierModel, self).__init__()

        # Number of continuous dimensions
        self.num_objects_ = hyperparams["num_objects"]
        self.flow_dim_ = 2 + 4*hyperparams["num_objects"]

        # Set up the permutation model
        permute_classes_continuous = []
        permute_classes_discrete = []
        for i in range(hyperparams["num_objects"]):
            permute_classes_continuous.append(np.array([2+4*i, 3+4*i, 4+4*i, 5+4*i]))
            permute_classes_discrete.append(np.array([i]))
        permute_classes_continuous = np.array(permute_classes_continuous)
        permute_classes_discrete = np.array(permute_classes_discrete)

        self.permuter_ = None
        if hyperparams["permutation"] == 'stochastic':
            self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)

        # Compute the factors required for the encoding of categorical labels     
        dropout_factors = torch.ones(hyperparams["num_objects"]).long()
        for i in range(1, hyperparams["num_objects"]):
            dropout_factors[i] = dropout_factors[i-1]*2
        self.register_buffer("dropout_factors_", dropout_factors)

        # -------------------- Load the dropout probs dict -------------------        
        with open(hyperparams["data_dir"] + "/dicts/dropout_dict_chan_" + hyperparams['channel'] + "_" + hyperparams['permutation'] + "_num_objects_" + str(hyperparams["num_objects"]) + ".yaml") as dropout_dict_data:
            dropout_dict = yaml.load(dropout_dict_data, Loader=yaml.Loader)

        log_probs_dropout = torch.zeros(2**hyperparams["num_objects"])

        # Fill in the maps between index and category
        for dropout in dropout_dict:
            if dropout != "tot_samples":
                log_probs_dropout[dropout] = dropout_dict[dropout]
    
        total_samples = dropout_dict["tot_samples"]

        log_probs_dropout = torch.log(log_probs_dropout/total_samples)

        # Set the embedding
        self.embedding_ = nn.Embedding(2**hyperparams["num_objects"], hyperparams["embedding_size"])

        # Base distribution
        self.base_dist_ = StandardMixtureNormal(self.flow_dim_, 2**hyperparams["num_objects"])
        self.base_dist_.categorical_log_probs_ = log_probs_dropout

        # Set up the flow model
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                context_features=hyperparams["embedding_size"],
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="linear", 
                tail_bound = 5
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

        # Set up a transform that only contains the permutations of the flow
        # Used for propagating the masking operation in the inverse direction
        permute_transforms = []
        for transform in flow_transforms:
            if isinstance(transform, RandomPermutation):
                permute_transforms.append(transform)
        self.composite_permute_transform_ = CompositeTransform(permute_transforms)

        # -------------------- Classifier model -------------------
        classifier_layers = []
        classifier_layers.append(nn.Linear(self.flow_dim_, hyperparams['classifier_size']))
        classifier_layers.append(nn.ReLU())
        for _ in range(hyperparams['classifier_layers']-1):
            classifier_layers.append(nn.Linear(hyperparams['classifier_size'], hyperparams['classifier_size']))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Linear(hyperparams['classifier_size'], 7*hyperparams['num_objects']))
        self.classifier_ = nn.Sequential(*classifier_layers)


    def log_prob_flow(self, inputs_continuous, inputs_discrete, permute=True):
        # Permute
        if permute and self.permuter_ is not None:
            inputs_continuous, inputs_discrete, _ = self.permuter_.forward(inputs_continuous, inputs_discrete)

        # Find dropped objects
        dropped_objects = inputs_discrete == 0

        # Get the dropout label
        dropout_encoded = torch.sum(self.dropout_factors_*(inputs_discrete != 0), dim=-1).squeeze()

        # Pass throught the flow
        inputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous, context=self.embedding_(dropout_encoded.clone()))

        # Evaluate the base distribution
        base_log_prob = self.base_dist_.log_prob_joint(inputs_continuous, dropout_encoded)

        return transform_log_prob + base_log_prob

    def log_prob_classifier(self, inputs_continuous, inputs_discrete, permute=True):
        # Permute
        if permute and self.permuter_ is not None:
            inputs_continuous, inputs_discrete, _ = self.permuter_.forward(inputs_continuous, inputs_discrete)

        # Replace nans by zeros
        inputs_continuous = torch.nan_to_num(inputs_continuous, 0.)

        # Get logits, reshape and logsoftmax
        classifier_output = F.softmax(self.classifier_(inputs_continuous).reshape(inputs_continuous.shape[0], self.num_objects_, 7), -1)

        # Find the dropped objects
        dropped_objects = inputs_discrete == 0

        # Replace all logprobs from dropped objects by a 0
        classifier_output = torch.masked_fill(classifier_output, dropped_objects.unsqueeze(-1), 0.)

        # The categories are 0-6, so subtract 1 then set all -1's to zeros
        inputs_discrete = inputs_discrete - 1
        inputs_discrete[inputs_discrete == -1] = 0

        # Select the correct logprobs and sum over the objects
        return torch.sum(torch.gather(classifier_output, -1, inputs_discrete.unsqueeze(-1)).squeeze(), dim=-1)
        
    def log_prob(self, inputs_continuous, inputs_discrete):
        # Permute
        if self.permuter_ is not None:
            inputs_continuous, inputs_discrete, _ = self.permuter_.forward(inputs_continuous, inputs_discrete)

        return self.log_prob_flow(inputs_continuous.clone(), inputs_discrete.clone(), False) + self.log_prob_classifier(inputs_continuous, inputs_discrete, False)