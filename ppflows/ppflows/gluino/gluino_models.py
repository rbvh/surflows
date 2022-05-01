import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import yaml

from ppflows.rqs_flow.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from ppflows.rqs_flow.transforms import RandomPermutation, CompositeTransform
from ppflows.gluino.gluino_auxiliary import GluinoCategoricalEncoder, angle_ordered_permutation

from ppflows.distributions import BetaMixtureBox, UniformBox, UniformMixtureBox
from ppflows.permuters import IteratedPermutation, StochasticPermutation, SortPermutation
from ppflows.argmax import ArgmaxUniform, ArgmaxFlow
from ppflows.dequantization import DequantizationUniform, DequantizationFlow
from ppflows.rqs_flow.transforms import Permutation


'''
Model that includes permutations and discrete features through a surjective layer
'''
class GluinoModel(nn.Module):
    def __init__(self, hyperparams):
        super(GluinoModel, self).__init__()

        # Gluino encoder
        self.encoder_  = GluinoCategoricalEncoder()

        # Number of continuous dimensions
        dim_continuous = 8

        # ------------------------- Set up the discrete model -------------------------
        self.num_categories_per_dim_ = torch.tensor([64,120])
        
        self.discrete_layer_ = None
        if "discrete_mode" in hyperparams:
            assert(hyperparams["permutation"] == "stochastic" or hyperparams["permutation"] == "ordered")

            # Option 1: argmax with uniform dequantization
            if hyperparams["discrete_mode"] == "uniform_argmax":
                self.discrete_layer_ = ArgmaxUniform(self.num_categories_per_dim_, dim_continuous)
            # Option 2: argmax with flow distribution dequantization
            if hyperparams["discrete_mode"] == "flow_argmax":
                self.discrete_layer_ = ArgmaxFlow(self.num_categories_per_dim_, dim_continuous)
            # Option 3: dequantization with uniform dequantization
            if hyperparams["discrete_mode"] == "uniform_dequantization":
                self.discrete_layer_ = DequantizationUniform(self.num_categories_per_dim_, dim_continuous)
            # Option 4: dequantization with flow dequantization
            if hyperparams["discrete_mode"] == "flow_dequantization":
                self.discrete_layer_ = DequantizationFlow(self.num_categories_per_dim_, dim_continuous)

        # Set the flow dimension
        if self.discrete_layer_ is not None:
            self.flow_dim_ = self.discrete_layer_.num_flow_dimensions()
        else:
            self.flow_dim_ = dim_continuous

        # ------------------------- Set up the permutation model -------------------------
        # Only include discrete permutation if we are doing a discrete model
        permute_classes_continuous = np.array([[0,1], [2,3], [4,5], [6,7]])
        if self.discrete_layer_ is not None:
            permute_classes_discrete = np.array([[2,7],[3,8],[4,9],[5,10]])
        else:
            permute_classes_discrete = None

        # Default corresponds with no permutation
        self.permuter_ = None
        if "permutation" in hyperparams:
            if hyperparams["permutation"] == "stochastic":
                self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)
            if hyperparams["permutation"] == "ordered":
                self.permuter_ = SortPermutation(angle_ordered_permutation, permute_classes_continuous, permute_classes_discrete)
        
        # ------------------------- Set up the flow model -------------------------
        self.base_dist_ = UniformBox(self.flow_dim_)

        # Set up the flow distribution
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="restricted",
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

    # ----- Forward and backward passes -----
    def _forward(self, inputs_continuous, inputs_discrete, permute=True):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Permute
        permute_log_prob = 0
        if permute and self.permuter_ is not None:
            if self.discrete_layer_ is not None:
                inputs_continuous, inputs_discrete, permute_log_prob = self.permuter_.forward(inputs_continuous, inputs_discrete)
            else:
                inputs_continuous, _, permute_log_prob = self.permuter_.forward(inputs_continuous, None)
        total_log_prob += permute_log_prob

        # Discrete layer
        if self.discrete_layer_ is not None:
            # Encoder 
            inputs_discrete = self.encoder_.encode(inputs_discrete)

            # Discrete layer
            inputs_continuous, discrete_log_prob = self.discrete_layer_.forward(inputs_continuous, inputs_discrete)
            total_log_prob += discrete_log_prob

        # Pass through the flow transform
        inputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous)
        total_log_prob += transform_log_prob

        return inputs_continuous, total_log_prob

    def _inverse(self, inputs_continuous, permute=True):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Flow transforms
        inputs_continuous, transform_log_prob = self.composite_flow_transform_.inverse(inputs_continuous)
        total_log_prob += transform_log_prob

        # Apply discrete layer
        inputs_discrete = None
        if self.discrete_layer_ != None:
            inputs_continuous, inputs_discrete, discrete_log_prob = self.discrete_layer_.inverse(inputs_continuous)
            total_log_prob += discrete_log_prob

            # Kick out samples with non-viable category
            category_mask = ~torch.any(inputs_discrete > self.num_categories_per_dim_, dim=-1)
            inputs_continuous = inputs_continuous[category_mask]
            inputs_discrete = inputs_discrete[category_mask]
            total_log_prob = total_log_prob[category_mask]
        
        # Decode 
        if inputs_discrete != None:
            inputs_discrete = self.encoder_.decode(inputs_discrete)
        
        # Permute
        permute_log_prob = 0
        if permute and self.permuter_ is not None:
            if self.discrete_layer_ is not None:
                inputs_continuous, inputs_discrete, permute_log_prob = self.permuter_.inverse(inputs_continuous, inputs_discrete)
            else:
                inputs_continuous, _, permute_log_prob = self.permuter_.inverse(inputs_continuous, None)
        total_log_prob += permute_log_prob

        return inputs_continuous, inputs_discrete, total_log_prob

    # ----- Sampling and likelihood evaluation -----
    def log_prob(self, inputs_continuous, inputs_discrete=None, permute=True):
        inputs_continuous, total_log_prob = self._forward(inputs_continuous, inputs_discrete, permute=permute)

        # Evaluate the base distribution
        total_log_prob += self.base_dist_.log_prob(inputs_continuous)

        return total_log_prob

    def sample(self, num_samples, batch_size=None, permute=True):
        if batch_size is None:
            return self._sample(num_samples, permute=permute)
        
        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples_continuous = []
            samples_discrete = []

            for _ in range(num_batches):
                sample_continuous, sample_discrete, log_probs = self._sample(batch_size, permute=permute)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
            
            if num_leftover > 0:
                sample_continuous, sample_discrete, log_probs = self._sample(num_leftover, permute=permute)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)

        if self.discrete_layer_ is not None:
            return torch.cat(samples_continuous, dim=0), torch.cat(samples_discrete, dim=0)
        else:
            return torch.cat(samples_continuous, dim=0), None

    def _sample(self, num_samples, permute):
        # Sample from the base distribution
        try:
            samples_continuous = self.base_dist_.sample(num_samples)
            samples_continuous, samples_discrete, _ = self._inverse(samples_continuous, permute=permute)

            return samples_continuous, samples_discrete

        except ValueError as error:
            return self._sample(num_samples, permute)

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
'''
Model that includes discrete features through a mixture prior
'''
class MixtureGluinoModel(nn.Module):
    def __init__(self, hyperparams):
        super(MixtureGluinoModel, self).__init__()

        # Gluino encoder
        self.encoder_  = GluinoCategoricalEncoder()

        # Number of continuous dimensions
        self.flow_dim_ = 8

        # Number of categories per dimension
        self.num_categories_per_dim_ = torch.tensor([64,120])
        self.total_num_categories_ = torch.prod(self.num_categories_per_dim_).item()

        # ------------------------ Embedding for categories -----------------------
        self.embedding_size_ = 64
        self.embedding_ = nn.Embedding(self.total_num_categories_, self.embedding_size_)
        
        # ------------------------- Set up the permutation model -------------------------
        # Only include discrete permutation if we are doing a discrete model
        permute_classes_continuous = np.array([[0,1], [2,3], [4,5], [6,7]])
        permute_classes_discrete    = np.array([[2,7],[3,8],[4,9],[5,10]])

        self.permuter_ = None
        if "permutation" in hyperparams:
            if hyperparams["permutation"] == "stochastic":
                self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)
                self.pretraining_permuter_ = IteratedPermutation(permute_classes_continuous, permute_classes_discrete)
            elif hyperparams["permutation"] == "ordered":
                self.permuter_ = SortPermutation(angle_ordered_permutation, permute_classes_continuous, permute_classes_discrete)
                        
        # ------------------------- Set up the flow model -------------------------
        self.base_dist_ = UniformMixtureBox(self.flow_dim_, self.total_num_categories_)
        
        # Set up the flow distribution
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                context_features=self.embedding_size_,
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="restricted",
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

        # -------------------- Load the categorical probs dict -------------------
        log_probs_categorical = torch.zeros(self.total_num_categories_).long()

        with open(hyperparams["data_dir"] + "/dict.yaml") as category_dict_data:
            category_dict = yaml.safe_load(category_dict_data)

        for key in category_dict:
            if key != "tot_samples":
                log_probs_categorical[key] = category_dict[key]
        total_samples = category_dict["tot_samples"]
        log_probs_categorical = torch.log(log_probs_categorical.float()/total_samples)
        
        self.base_dist_.categorical_log_probs_ = log_probs_categorical
        
    # ----- Forward and backward passes -----
    def _forward(self, inputs_continuous, inputs_discrete, permute=True):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Permute
        if permute:
            inputs_continuous, inputs_discrete, permute_log_prob = self.permuter_.forward(inputs_continuous, inputs_discrete)
            total_log_prob += permute_log_prob

        # Encoder
        inputs_discrete = self.encoder_.encode(inputs_discrete)
        inputs_discrete = inputs_discrete[:,0] + inputs_discrete[:,1]*self.num_categories_per_dim_[0]

        # Pass through the flow transform
        inputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous, context=self.embedding_(inputs_discrete))
        total_log_prob += transform_log_prob

        return inputs_continuous, inputs_discrete, total_log_prob

    def _inverse(self, inputs_continuous, inputs_discrete, permute=True):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Flow transforms
        inputs_continuous, transform_log_prob = self.composite_flow_transform_.inverse(inputs_continuous, context=self.embedding_(inputs_discrete))
        total_log_prob += transform_log_prob
        
        # Decode 
        inputs_discrete = torch.stack((inputs_discrete % self.num_categories_per_dim_[0], inputs_discrete / self.num_categories_per_dim_[0]), dim=1)
        inputs_discrete = self.encoder_.decode(inputs_discrete)

        # Permute
        if permute and self.permuter_ != None:
            inputs_continuous, inputs_discrete, permute_log_prob = self.permuter_.inverse(inputs_continuous, inputs_discrete)
            total_log_prob += permute_log_prob

        return inputs_continuous, inputs_discrete, total_log_prob

    # ----- Sampling and likelihood evaluation -----
    def log_prob_conditional(self, inputs_continuous, inputs_discrete, permute=True):
        inputs_continuous, inputs_discrete, total_log_prob = self._forward(inputs_continuous, inputs_discrete, permute=permute)

        # Evaluate the base distribution
        total_log_prob += self.base_dist_.log_prob_conditional(inputs_continuous, inputs_discrete)

        return total_log_prob

    def log_prob(self, inputs_continuous, inputs_discrete, permute=True):
        inputs_continuous, inputs_discrete, total_log_prob = self._forward(inputs_continuous, inputs_discrete, permute=permute)

        # Evaluate the base distribution
        total_log_prob += self.base_dist_.log_prob_joint(inputs_continuous, inputs_discrete)

        return total_log_prob

    def sample(self, num_samples, batch_size=None, permute=True):
        if batch_size is None:
            return self._sample(num_samples, permute=permute)
        
        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples_continuous = []
            samples_discrete = []

            for _ in range(num_batches):
                sample_continuous, sample_discrete = self._sample(batch_size, permute=permute)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
            
            if num_leftover > 0:
                sample_continuous, sample_discrete = self._sample(num_leftover, permute=permute)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)

        return torch.cat(samples_continuous, dim=0), torch.cat(samples_discrete, dim=0)
    
    def sample_conditional(self, inputs_discrete, batch_size=None, permute=True):
        if batch_size is None:
            return self._sample_conditional(inputs_discrete, permute=permute)

        else:
            num_batches = inputs_discrete.shape[0] // batch_size
            num_leftover = inputs_discrete.shape[0] % batch_size
            samples_continuous = []
            samples_discrete = []

            for i in range(num_batches):
                sample_continuous, sample_discrete = self._sample_conditional(inputs_discrete[i*batch_size : (i+1)*batch_size], permute=permute)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
            
            if num_leftover > 0:
                sample_continuous, sample_discrete = self._sample_conditional(inputs_discrete[num_batches*batch_size : ], permute=permute)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
        
        return torch.cat(samples_continuous, dim=0), torch.cat(samples_discrete, dim=0)

    def _sample(self, num_samples, permute=True):
        try:
            samples_continuous, samples_discrete = self.base_dist_.sample(num_samples)
            samples_continuous, samples_discrete, _ = self._inverse(samples_continuous, samples_discrete, permute=permute)
            return samples_continuous, samples_discrete
        except ValueError as error:
            return self._sample(num_samples, permute)

    def _sample_conditional(self, inputs_discrete, permute=True):
        # This function expects unencoded input -> have to encode
        inputs_discrete = self.encoder_.encode_discrete(inputs_discrete)
        inputs_discrete = inputs_discrete[:,0] + inputs_discrete[:,1]*self.num_categories_per_dim_[0]

        samples_continuous = self.base_dist_.sample_conditional(inputs_discrete)

        samples_continuous, samples_discrete, _ = self._inverse(samples_continuous, inputs_discrete, permute=permute)
        return samples_continuous, samples_discrete

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
'''
Model that mixes 2gluino and 4gluino samples through a mixture prior
'''
class DropoutGluinoModel(nn.Module):
    def __init__(self, hyperparams):
        super(DropoutGluinoModel, self).__init__()

        # Number of continuous dimensions
        self.flow_dim_ = 8
        
        # ------------------------- Set up the permutation model -------------------------
        # Only include discrete permutation if we are doing a discrete model
        permute_classes_continuous = np.array([[0,1], [2,3], [4,5], [6,7]])

        # Default corresponds with no permutation
        self.permuter_ = None
        if "permutation" in hyperparams:
            if hyperparams["permutation"] == "stochastic":
                self.permuter_ = StochasticPermutation(permute_classes_continuous, None)
            if hyperparams["permutation"] == "ordered":
                self.permuter_ = SortPermutation(angle_ordered_permutation, permute_classes_continuous, None)

        # ----------------- Embedding for dropout categories ----------------        
        self.embedding_ = nn.Embedding(2, 2)

        # ------------------------- Set up the flow model -------------------------
        self.base_dist_ = UniformMixtureBox(self.flow_dim_, 2)
        # These are the log of the normalized cross sections
        self.base_dist_.categorical_log_probs_ = torch.tensor([-10.1248937579, -0.00004007035])

        # Set up the flow distribution
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                context_features=2,
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="restricted",
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)  

        # We need to figure out where dropped indices end up after permutations
        dropout_indices = torch.ones(1, 8)
        dropout_indices[0,0] = 0
        dropout_indices[0,1] = 0
        for transform in self.composite_flow_transform_._transforms:
            if isinstance(transform, RandomPermutation):
                dropout_indices, _ = transform(dropout_indices)
        self.register_buffer("dropout_indices_", dropout_indices.squeeze().bool())
    
    # ----- Forward and backward passes -----
    def _forward(self, inputs_continuous, permute=True):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Determine the dropout category
        # 0 corresponds with no dropout
        # 1 corresponds with dropout
        inputs_discrete = torch.any(inputs_continuous == -1, dim=1).long()

        # Permute - Only acts on the 4 body states
        if permute and self.permuter_ != None:
            inputs_continuous[~inputs_discrete.bool()], _, permute_log_prob = self.permuter_.forward(inputs_continuous[~inputs_discrete.bool()], None)
            total_log_prob[~inputs_discrete.bool()] += permute_log_prob

        # Pass through the flow transform
        inputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous, context=self.embedding_(inputs_discrete))
        total_log_prob += transform_log_prob

        return inputs_continuous, inputs_discrete, total_log_prob

    def _inverse(self, inputs_continuous, inputs_discrete, permute=True):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Dropout on rows that have inputs_discrete = 1
        inputs_continuous[torch.nonzero(inputs_discrete), self.dropout_indices_] = -1

        # Flow transforms
        inputs_continuous, transform_log_prob = self.composite_flow_transform_.inverse(inputs_continuous, context=self.embedding_(inputs_discrete))
        total_log_prob += transform_log_prob

        if permute and self.permuter_ != None:
            inputs_continuous[~inputs_discrete.bool()], _, permute_log_prob = self.permuter_.inverse(inputs_continuous[~inputs_discrete.bool()], None)
            total_log_prob[~inputs_discrete.bool()] += permute_log_prob

        return inputs_continuous, total_log_prob

    # ----- Sampling and likelihood evaluation -----
    def log_prob_conditional(self, inputs_continuous, permute=True):
        inputs_continuous, inputs_discrete, total_log_prob = self._forward(inputs_continuous, permute=permute)

        total_log_prob += self.base_dist_.log_prob_conditional(inputs_continuous, inputs_discrete)
        
        return total_log_prob

    def log_prob(self, inputs_continuous, permute=True):
        inputs_continuous, inputs_discrete, total_log_prob = self._forward(inputs_continuous, permute=permute)

        total_log_prob += self.base_dist_.log_prob_joint(inputs_continuous, inputs_discrete)

        return total_log_prob

    def sample(self, num_samples, batch_size=None, permute=True):
        if batch_size is None:
            return self._sample(num_samples, permute=permute)
        
        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples_continuous = []

            for _ in range(num_batches):
                sample_continuous = self._sample(batch_size, permute=permute)
                samples_continuous.append(sample_continuous)
                
            if num_leftover > 0:
                sample_continuous = self._sample(num_leftover, permute=permute)
                samples_continuous.append(sample_continuous)

        return torch.cat(samples_continuous, dim=0)
    
    def sample_conditional(self, inputs_discrete, batch_size=None, permute=True):
        if batch_size is None:
            return self._sample_conditional(inputs_discrete, permute=permute)

        else:
            num_batches = inputs_discrete.shape[0] // batch_size
            num_leftover = inputs_discrete.shape[0] % batch_size
            samples_continuous = []

            for i in range(num_batches):
                sample_continuous = self._sample_conditional(inputs_discrete[i*batch_size : (i+1)*batch_size], permute=permute)
                samples_continuous.append(sample_continuous)
            
            if num_leftover > 0:
                sample_continuous = self._sample_conditional(inputs_discrete[num_batches*batch_size : ])
                samples_continuous.append(sample_continuous)
        
        return torch.cat(samples_continuous, dim=0)

    def _sample(self, num_samples, permute=True):
        try:
            samples_continuous, samples_discrete = self.base_dist_.sample(num_samples)
            samples_continuous, _ = self._inverse(samples_continuous, samples_discrete, permute=permute)

            return samples_continuous

        except ValueError as error:
            return self._sample(num_samples, permute)

    def _sample_conditional(self, inputs_discrete, permute=True):
        try:
            samples_continuous = self.base_dist_.sample_conditional(inputs_discrete)
            samples_continuous, log_probs = self._inverse(samples_continuous, inputs_discrete, permute=permute)

            return samples_continuous, log_probs

        except ValueError as error:
            return self._sample(inputs_discrete, permute)

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
'''
Simple classifier for discrete models
'''
class DiscreteClassifier(nn.Module):
    def __init__(self, hyperparams):
        super(DiscreteClassifier, self).__init__()

        # Gluino encoder
        self.encoder_  = GluinoCategoricalEncoder()

        # Number of continuous dimensions
        self.continuous_dim_ = 8

        # Number of categories per dimension
        self.num_categories_per_dim_ = torch.tensor([64, 120])
        self.total_num_categories_ = torch.prod(self.num_categories_per_dim_).item()
        
        # ------------------------- Set up the permutation model -------------------------
        # Only include discrete permutation if we are doing a discrete model
        permute_classes_continuous = np.array([[0,1], [2,3], [4,5], [6,7]])
        permute_classes_discrete    = np.array([[2,7],[3,8],[4,9],[5,10]])

        self.permuter_ = None
        if "permutation" in hyperparams:
            if hyperparams["permutation"] == "stochastic":
                self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)
            elif hyperparams["permutation"] == "ordered":
                self.permuter_ = SortPermutation(angle_ordered_permutation, permute_classes_continuous, permute_classes_discrete)

        classifier_layers = []
        classifier_layers.append(nn.Linear(self.continuous_dim_, hyperparams['classifier_size']))
        classifier_layers.append(nn.ReLU())
        for _ in range(hyperparams['classifier_layers']-1):
            classifier_layers.append(nn.Linear(hyperparams['classifier_size'], hyperparams['classifier_size']))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Linear(hyperparams['classifier_size'], self.num_categories_per_dim_[0]*self.num_categories_per_dim_[1]))
        self.classifier = nn.Sequential(*classifier_layers)

    '''
    Returns unnormalized weights for all classes, and includes a permutation
    '''
    def forward_train(self, inputs_continuous, inputs_discrete):
        # Permute
        inputs_continuous, inputs_discrete, permute_log_prob = self.permuter_.forward(inputs_continuous, inputs_discrete)

        # Encoder
        inputs_discrete = self.encoder_.encode(inputs_discrete)
        inputs_discrete = inputs_discrete[:,0] + inputs_discrete[:,1]*self.num_categories_per_dim_[0]

        # Pass through classifier
        output = self.classifier(inputs_continuous)

        return output, inputs_discrete

    '''
    Returns categorical log probs for classes given continuous input
    '''
    def log_prob(self, inputs_continuous):
        return nn.functional.log_softmax(self.classifier(inputs_continuous), dim=1)
    
    def logit(self, inputs_continuous):
        return self.classifier(inputs_continuous)

class ClassifierGluinoModel(nn.Module):
    def __init__(self, hyperparams):
        super(ClassifierGluinoModel, self).__init__()

        # Gluino encoder
        self.encoder_  = GluinoCategoricalEncoder()

        self.flow_dim_ = 8

        # Number of categories per dimension
        self.num_categories_per_dim_ = torch.tensor([64, 120])
        self.total_num_categories_ = torch.prod(self.num_categories_per_dim_).item()

        # ------------------------- Set up the permutation model -------------------------
        # Only include discrete permutation if we are doing a discrete model
        permute_classes_continuous = np.array([[0,1], [2,3], [4,5], [6,7]])
        permute_classes_discrete = np.array([[2,7],[3,8],[4,9],[5,10]])
        
        # Default corresponds with no permutation
        self.permuter_ = None
        if "permutation" in hyperparams:
            if hyperparams["permutation"] == "stochastic":
                self.permuter_ = StochasticPermutation(permute_classes_continuous, permute_classes_discrete)
            if hyperparams["permutation"] == "ordered":
                self.permuter_ = SortPermutation(angle_ordered_permutation, permute_classes_continuous, permute_classes_discrete)
        
        # ------------------------- Set up the flow model -------------------------
        self.base_dist_ = UniformBox(self.flow_dim_)

        # Set up the flow distribution
        flow_transforms = []
        for _ in range(hyperparams["n_flow_layers"]):
            flow_transforms.append(RandomPermutation(features=self.flow_dim_))
            flow_transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.flow_dim_, 
                hidden_features=hyperparams["n_made_units_per_dim"]*self.flow_dim_,
                num_bins=hyperparams["n_RQS_knots"],
                num_blocks=hyperparams["n_made_layers"],
                tails="restricted",
            ))
        self.composite_flow_transform_ = CompositeTransform(flow_transforms)

        # ------------------------- Set up the classifier -------------------------
        self.classifier = DiscreteClassifier(hyperparams)

    def load(self, flow_state_dict_path, classifier_state_dict_path, device):
        flow_state_dict = torch.load(flow_state_dict_path, map_location=torch.device(device))

        base_state_dict = self.base_dist_.state_dict()
        encoder_state_dict = self.encoder_.state_dict()
        composite_flow_dict = self.composite_flow_transform_.state_dict()

        for key in flow_state_dict:
            new_key = key.split(".", 1)[1]
            if 'encoder_' in key:
                encoder_state_dict[new_key] = flow_state_dict[key]
            elif 'base_dist_' in key:
                base_state_dict[new_key] = flow_state_dict[key]
            elif 'composite_flow_transform_' in key:
                composite_flow_dict[new_key] = flow_state_dict[key]

        self.base_dist_.load_state_dict(base_state_dict)
        self.encoder_.load_state_dict(encoder_state_dict)
        self.composite_flow_transform_.load_state_dict(composite_flow_dict)

        self.classifier.load_state_dict(torch.load(classifier_state_dict_path, map_location=torch.device(device)))

    # ----- Forward and backward passes -----
    def _forward(self, inputs_continuous, inputs_discrete):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Permute
        inputs_continuous, inputs_discrete, permute_log_prob = self.permuter_.forward(inputs_continuous, inputs_discrete)
        total_log_prob += permute_log_prob

        # Encode
        inputs_discrete = self.encoder_.encode(inputs_discrete)
        inputs_discrete = inputs_discrete[:,0] + inputs_discrete[:,1]*self.num_categories_per_dim_[0]

        # Get classifier log probs
        classifier_log_probs = self.classifier.log_prob(inputs_continuous)
        classifier_log_probs = classifier_log_probs[torch.arange(classifier_log_probs.shape[0]), inputs_discrete]
        
        total_log_prob += classifier_log_probs

        # Pass through the flow transform
        inputs_continuous, transform_log_prob = self.composite_flow_transform_(inputs_continuous)
        total_log_prob += transform_log_prob

        return inputs_continuous, total_log_prob

    def _inverse(self, inputs_continuous):
        total_log_prob = torch.zeros(inputs_continuous.shape[0], device=inputs_continuous.device)

        # Flow transforms
        inputs_continuous, transform_log_prob = self.composite_flow_transform_.inverse(inputs_continuous)
        total_log_prob += transform_log_prob

        # Permute
        inputs_continuous, _, permute_log_prob = self.permuter_.inverse(inputs_continuous, None)
        total_log_prob += permute_log_prob

        # Sample from classifier
        classifier_output = self.classifier.log_prob(inputs_continuous)
        categorical_noise = torch.rand(inputs_continuous.shape[0], self.total_num_categories_, device=inputs_continuous.device)
        categorical_samples = torch.argmax(classifier_output - torch.log(-torch.log(categorical_noise)), dim=-1)

        total_log_prob += classifier_output[torch.arange(inputs_continuous.shape[0]), categorical_samples]

        # Decode
        categorical_samples = torch.stack((categorical_samples % self.num_categories_per_dim_[0], categorical_samples / self.num_categories_per_dim_[0]), dim=1)
        categorical_samples = self.encoder_.decode(categorical_samples)

        return inputs_continuous, categorical_samples, total_log_prob

    # ----- Sampling and likelihood evaluation -----
    def log_prob(self, inputs_continuous, inputs_discrete=None):
        inputs_continuous, total_log_prob = self._forward(inputs_continuous, inputs_discrete)

        # Evaluate the base distribution
        total_log_prob += self.base_dist_.log_prob(inputs_continuous)

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
                sample_continuous, sample_discrete, log_probs = self._sample(batch_size)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)
            
            if num_leftover > 0:
                sample_continuous, sample_discrete, log_probs = self._sample(num_leftover)
                samples_continuous.append(sample_continuous)
                samples_discrete.append(sample_discrete)

        if self.discrete_layer_ is not None:
            return torch.cat(samples_continuous, dim=0), torch.cat(samples_discrete, dim=0)
        else:
            return torch.cat(samples_continuous, dim=0), None

    def _sample(self, num_samples):
        # Sample from the base distribution
        try:
            samples_continuous = self.base_dist_.sample(num_samples)
            samples_continuous, samples_discrete, _ = self._inverse(samples_continuous)

            return samples_continuous, samples_discrete

        except ValueError as error:
            return self._sample(num_samples)