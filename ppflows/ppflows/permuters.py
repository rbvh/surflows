import torch
from torch import nn
import numpy as np
import math
import itertools
"""
Class that stochastically permutes data.
Supports data that is split into two tensors

Note: This class takes np arrays as initialization
Permutation always occurs in the first index
"""

class PermutationBase(nn.Module):
    # The permute classes are 2d arrays with subsets of indices that may be permuted
    # That is, if permute_classes_1 = [[0,2],[1,3]], permutations are [0,2]<->[1,3]
    def __init__(self, permute_classes_1=None, permute_classes_2=None):
        super().__init__()

        # Make sure the classes are sensible

        # At least one of permute_classes must not be None
        assert permute_classes_1 is not None or permute_classes_2 is not None

        # Make sure that either is None or a 2d numpy array
        if permute_classes_1 is not None:
            assert type(permute_classes_1) is np.ndarray
            assert len(permute_classes_1.shape) == 2
            self.permute_classes_1 = torch.tensor(permute_classes_1)
        else:
            self.permute_classes_1 = None

        if permute_classes_2 is not None:
            assert type(permute_classes_2) is np.ndarray
            assert len(permute_classes_2.shape) == 2
            self.permute_classes_2 = torch.tensor(permute_classes_2)
        else:
            self.permute_classes_2 = None

        # If both are not none, they should be of the same size
        if permute_classes_1 is not None and permute_classes_2 is not None:
            # Number of classes should be the same
            assert permute_classes_1.shape[0] == permute_classes_2.shape[0]
        
        self.n_classes = permute_classes_1.shape[0] if permute_classes_1 is not None else permute_classes_2.shape[0]
    
    def permute(self, permutation_of_classes, inputs_1=None, inputs_2=None):
        outputs_1 = None
        outputs_2 = None

        if inputs_1 is not None:
            device = inputs_1.device
            batch_size = inputs_1.shape[0]

            permutation_1_mutable_only = self.permute_classes_1[permutation_of_classes].reshape(batch_size, torch.numel(self.permute_classes_1))
            permutation_1 = torch.arange(0, inputs_1.shape[1], device=inputs_1.device).unsqueeze(0).repeat(batch_size, 1)

            for i, el in enumerate(self.permute_classes_1.view(-1)):
                permutation_1[:,el] = permutation_1_mutable_only[:,i]

            for d in range(2, inputs_1.dim()):
                permutation_1 = permutation_1.unsqueeze(-1)

            permutation_1 = permutation_1.expand_as(inputs_1)
            outputs_1 = torch.gather(inputs_1, 1, permutation_1)

        if inputs_2 is not None:
            device = inputs_2.device
            batch_size = inputs_2.shape[0]
    
            permutation_2_mutable_only = self.permute_classes_2[permutation_of_classes].reshape(batch_size, torch.numel(self.permute_classes_2))
            permutation_2 = torch.arange(0, inputs_2.shape[1], device=inputs_2.device).unsqueeze(0).repeat(batch_size, 1)

            for i, el in enumerate(self.permute_classes_2.view(-1)):
                permutation_2[:,el] = permutation_2_mutable_only[:,i]
            
            for d in range(2, inputs_2.dim()):
                permutation_2 = permutation_2.unsqueeze(-1)

            permutation_2 = permutation_2.expand_as(inputs_2)
            outputs_2 = torch.gather(inputs_2, 1, permutation_2)

        return outputs_1, outputs_2

    def forward(self, inputs_1, inputs_2):
        raise ValueError('Tried to call forward of PermutationBase')

    def inverse(self, inputs_1, inputs_2):
        raise ValueError('Tried to call inverse of PermutationBase')

class StochasticPermutation(PermutationBase):
    def __init__(self, permute_classes_1=None, permute_classes_2=None):
        super().__init__(permute_classes_1=permute_classes_1, permute_classes_2=permute_classes_2)
    
    def forward(self, inputs_1, inputs_2):
        if inputs_1 is not None:
            device = inputs_1.device
            batch_size = inputs_1.shape[0]
        elif inputs_2 is not None:
            device = inputs_2.device
            batch_size = inputs_2.shape[0]
        else:
            return None, None, None

        rand = torch.rand(batch_size, self.n_classes, device=device)
        perm = rand.argsort(dim=1)
        inputs_1, inputs_2 = self.permute(perm, inputs_1, inputs_2)

        return inputs_1, inputs_2, torch.zeros(batch_size, device=device)

    def inverse(self, inputs_1, inputs_2):
        if inputs_1 is not None:
            device = inputs_1.device
            batch_size = inputs_1.shape[0]
        elif inputs_2 is not None:
            device = inputs_2.device
            batch_size = inputs_1.shape[0]
        else:
            return None, None, None

        rand = torch.rand(batch_size, self.n_classes, device=device)
        perm = rand.argsort(dim=1)
        inputs_1, inputs_2 = self.permute(perm, inputs_1, inputs_2)
        return inputs_1, inputs_2, torch.zeros(batch_size, device=device)

'''
Permutation that runs through options, used for pretraining of the discrete model
'''
class IteratedPermutation(PermutationBase):
    def __init__(self, permute_classes_1=None, permute_classes_2=None):
        super().__init__(permute_classes_1=permute_classes_1, permute_classes_2=permute_classes_2)

        self.counter_ = 0
        n_elements = permute_classes_1.shape[0] if permute_classes_1 is not None else permute_classes_2.shape[0]
    
        self.register_buffer("permutations_", torch.tensor(list(itertools.permutations(np.arange(n_elements)))))
    
    def forward(self, inputs_1, inputs_2):
        assert self.counter_ < self.permutations_.shape[0]
        
        if inputs_1 is not None:
            device = inputs_1.device
            batch_size = inputs_1.shape[0]
        elif inputs_2 is not None:
            device = inputs_2.device
            batch_size = inputs_2.shape[0]
        else:
            return None, None, None

        # Permute
        perm = self.permutations_[self.counter_].unsqueeze(0).repeat(batch_size, 1)
        inputs_1, inputs_2 = self.permute(perm, inputs_1, inputs_2)
        
        return inputs_1, inputs_2, torch.zeros(batch_size, device=device)
        
    def inverse(self, inputs_1, inputs_2):
        raise ValueError('Inverse should not be called on IteratedPermutation')

    # Raise counter by 1
    def next(self):
        self.counter_ += 1

        return self.counter_ < self.permutations_.shape[0]

'''
Permutation strategy that orders the inputs according to some metric in the forward direction.
The inverse direction permutes uniformly
'''

class SortPermutation(PermutationBase):
    def __init__(self, sorter, permute_classes_1=None, permute_classes_2=None):
        super().__init__(permute_classes_1=permute_classes_1, permute_classes_2=permute_classes_2)

        self.sorter = sorter

        # Log prob associated with sorting operation
        self._log_prob = math.log(math.factorial(permute_classes_1.shape[0]))

    def forward(self, inputs_1, inputs_2):
        if inputs_1 is not None:
            device = inputs_1.device
        elif inputs_2 is not None:
            device = inputs_2.device
        else:
            return None, None, None

        perm = self.sorter(inputs_1, inputs_2)
        inputs_1, inputs_2 = self.permute(perm, inputs_1, inputs_2)

        return inputs_1, inputs_2, -torch.ones(inputs_1.shape[0], device=device)*self._log_prob
            
    def inverse(self, inputs_1, inputs_2):
        if inputs_1 is not None:
            device = inputs_1.device
            batch_size = inputs_1.shape[0]
        elif inputs_2 is not None:
            device = inputs_2.device
            batch_size = inputs_2.shape[0]
        else:
            return None, None, None

        rand = torch.rand(batch_size, self.n_classes, device=device)
        perm = rand.argsort(dim=1)
        inputs_1, inputs_2 = self.permute(perm, inputs_1, inputs_2)

        return inputs_1, inputs_2, torch.ones(inputs_1.shape[0], device=device)*self._log_prob