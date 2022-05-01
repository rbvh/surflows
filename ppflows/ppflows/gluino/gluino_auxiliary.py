"""
Discrete encoder for gg->4gluino
Uses Lehmer codes for colour permutations
https://medium.com/@benjamin.botto/sequentially-indexing-permutations-a-linear-algorithm-for-computing-lexicographic-rank-a22220ffd6e3
"""
import torch
from torch import nn
import math
import pandas as pd
import os
import numpy as np
import yaml

from ppflows.file_lock import FileLock
from ppflows.permuters import IteratedPermutation

class GluinoCategoricalEncoder(nn.Module):
    def __init__(self, cont_size=16, hel_size=6, col_size=5):
        super().__init__()
        
        self.cont_size = cont_size
        self.hel_size = hel_size
        self.col_size = col_size

        # Set up table of factors for helicity data
        hel_factors = torch.ones(hel_size, dtype=torch.int)
        for i in range(hel_size-2, -1, -1):
            hel_factors[i] = hel_factors[i+1]*2
        self.register_buffer("hel_factors_", hel_factors)

        # Set up table of factorials for colour data
        col_factors = torch.ones(col_size, dtype=torch.int)
        for i in range(col_size-2, -1, -1):
            col_factors[i] = col_factors[i+1]*(col_size - 1 - i)
        self.register_buffer("col_factors_", col_factors)

    '''
    Encoder function
    '''
    # Encode discrete into 2 sets of numbers
    def encode(self, data_disc):
        # Split up into helicity data and colour data
        hel_data = data_disc[:,:self.hel_size]
        col_data = data_disc[:,self.hel_size:]

        # Clone helicity data and convert to binary ones and zeroes
        hel_binary = hel_data.clone()
        hel_binary[hel_binary == -1] = 0
        # Then find the number that corresponds with this binary sequence
        hel_code = torch.sum(hel_binary * self.hel_factors_, -1)

        # Clone the colour permutation because we're altering col_lehmer sequentially
        col_lehmer = col_data.clone() 
        for i in range(1,self.col_size-1):
            # Broadcasting such that lhs = (batch,size) and rhs = (batch,1)
            bools = col_data < col_data[:,i,None] 
            # Slice off such that we only have indices below the current one
            sliced_bools = bools[:,:i] 
            # Subtract one for every index < the current one that appeared earlier
            col_lehmer[:,i] -= torch.sum(sliced_bools, axis=1) 
        # Last entry is always zero
        col_lehmer[:,-1] = 0

        # Next, the lehmer sequence is converted to a number using a factorial number system
        col_code = torch.sum(col_lehmer * self.col_factors_, -1)

        return torch.stack((hel_code, col_code), dim=1)

    '''
    Decoder function
    '''

    def decode(self, categories):
        batch_size = categories.shape[0]

        # First decode the helicity data
        hel_code_running = categories[:,0]
        hel_data = torch.zeros((batch_size, self.hel_size), dtype=torch.int, device=categories.device)
        # Find binary representation
        for i in range(self.hel_size):
            # hel_data[:,i] = hel_code_running.floor_divide(self.hel_factors_[i])
            hel_data[:,i] = torch.div(hel_code_running, self.hel_factors_[i], rounding_mode='floor')
            hel_code_running = hel_code_running.remainder(self.hel_factors_[i])
        # Change zeroes to -1's
        hel_data[hel_data == 0] = -1

        # Now decode the colour code to the lehmer sequence
        col_code_running = categories[:,1]
        col_lehmer = torch.zeros((batch_size, self.col_size), dtype=torch.int, device=categories.device)

        for i in range(self.col_size-1):
            # col_lehmer[:,i] = col_code_running.floor_divide(self.col_factors_[i])
            col_lehmer[:,i] = torch.div(col_code_running, self.col_factors_[i], rounding_mode='floor')
            col_code_running = col_code_running.remainder(self.col_factors_[i])

        # Clone because the first index is the same
        col_data = col_lehmer.clone()

        # Start from 1, because the 0th index is always identical
        for i in range (1, self.col_size):
            # Start from zeroes in the new column
            col_data_now = torch.empty(batch_size, dtype=torch.int, device=categories.device).fill_(0)
            col_lehmer_now = col_lehmer[:,i]
            # Now loop
            while True:
                # Find indices where the current new_perm already appears in the previous indices
                bools_increment = torch.sum(col_data_now[:,None] == col_data[:,:i], axis=1) != False
                # If there are any such indices, increment and continue
                if torch.sum(bools_increment) != False:
                    col_data_now[bools_increment] += 1
                else:
                    # Increment every index that has remaining lehmer values
                    bools_increment = col_lehmer_now != 0
                    # If there are any such indices, increment new_perms_now, decrement lehmer_now and continue
                    if torch.sum(bools_increment) != False:
                        col_data_now[bools_increment] += 1
                        col_lehmer_now[bools_increment] -= 1
                    # Otherwise, we are done
                    else:
                        break

            # Insert into perm
            col_data[:,i] = col_data_now

        return torch.cat((hel_data, col_data), axis=1)
            
'''
Get the permutation that sorts the data by polar angle
'''
def angle_ordered_permutation(inputs_cont, inputs_disc):
    return torch.argsort(inputs_cont[:,::2], dim=1, descending=True)

'''
Rambo rescaling procedure: massless to massive
Assumes all masses are identical
'''
def massless_to_massive(p, m):
    batch_size = p.size()[0]
    sqrts = torch.sum(p, dim=1)[0,0]

    assert(sqrts > m*p.size()[1])

    p_abs2 = p[:,:,1]**2 + p[:,:,2]**2 + p[:,:,3]**2
    x = torch.ones(batch_size, 1, device=p.device)
    f = torch.ones(batch_size, 1, device=p.device)
    while (torch.max(f) > 1e-3):
        f = torch.sum(torch.sqrt(p_abs2 * x**2 + m**2), dim=1) - sqrts
        f_prime = torch.sum(p_abs2*x/torch.sqrt(p_abs2 * x**2 + m**2), dim=1)
        x -= torch.unsqueeze(f/f_prime, 1)

    p[:,:,1:4] *= torch.unsqueeze(x, 2)
    p[:,:,0] = torch.sqrt(p[:,:,1]**2 + p[:,:,2]**2 + p[:,:,3]**2 + m**2)

    # Fix cases where objects are empty, the above line sets the energy to the mass for those
    p[:,:,0][p[:,:,1] == 0.] = 0

    return p

'''
Rambo rescaling procedure: massive to massless
Assumes all masses are identical
'''
def massive_to_massless(p):
    sqrts = torch.sum(p, dim=1)[0,0]
    x = sqrts/torch.sum(torch.sqrt(p[:,:,1]**2 + p[:,:,2]**2 + p[:,:,3]**2), dim=1)

    p[:,:,1:4] *= x.unsqueeze(1).unsqueeze(1)
    p[:,:,0] = torch.sqrt(p[:,:,1]**2 + p[:,:,2]**2 + p[:,:,3]**2)
    
    return p

'''
To angle representation
'''
def to_angle_4_body(p):
    batch_size = p.shape[0]

    p = massive_to_massless(p)

    # Convert to angles
    angles = torch.empty(batch_size, 8, device=p.device)
    angles[:,::2] = (p[:,:,3]/p[:,:,0] + 1)/2
    angles[:,1::2] = torch.atan2(p[:,:,1], p[:,:,2])/2/np.pi + 0.5

    return angles

def to_angle_2_body(p):
    batch_size = p.shape[0]

    p = massive_to_massless(p)

    # Convert to angles 
    angles = torch.ones(batch_size, 8, device=p.device)*(-1)
    angles[:,0] = (p[:,0,3]/p[:,0,0] + 1)/2
    angles[:,1] = torch.atan2(p[:,0,1], p[:,0,2])/2/np.pi + 0.5

    return angles

def to_phase_space_4_body(angles, sqrts, m):
    batch_size = angles.shape[0]

    # To massless momenta
    costheta = angles[:,::2]*2 - 1
    sintheta = torch.sqrt(torch.max(torch.zeros_like(costheta, device=angles.device), 1. - costheta**2))
    phi = 2*(angles[:,1::2] - 0.5)*np.pi

    # Solve linear system to find energies
    A = torch.ones(batch_size, 4, 4, device=angles.device)
    A[:,1,:] = sintheta*torch.sin(phi)
    A[:,2,:] = sintheta*torch.cos(phi)
    A[:,3,:] = costheta

    b = torch.zeros(batch_size, 4, device=angles.device)
    b[:,0] = sqrts

    E = torch.linalg.solve(A, b)
    p = E.unsqueeze(2)*A.transpose(1,2)
    p = massless_to_massive(p, m)

    return p

def to_phase_space_2_body(angles, sqrts, m):
    batch_size = angles.shape[0]

    # To massless momenta
    costheta = angles[:,0]*2 - 1
    sintheta = torch.sqrt(torch.max(torch.zeros_like(costheta,device=angles.device), 1 - costheta**2))
    phi = 2*(angles[:,1] - 0.5)*np.pi

    p = torch.zeros(batch_size, 2, 4, device=angles.device)
    p[:,0,0] = sqrts/2
    p[:,1,0] = sqrts/2
    p[:,0,1] =  sintheta*torch.sin(phi)*sqrts/2
    p[:,1,1] = -sintheta*torch.sin(phi)*sqrts/2
    p[:,0,2] =  sintheta*torch.cos(phi)*sqrts/2
    p[:,1,2] = -sintheta*torch.cos(phi)*sqrts/2
    p[:,0,3] =  costheta*sqrts/2
    p[:,1,3] = -costheta*sqrts/2

    p = massless_to_massive(p, m)

    return p

'''
Data loader for gg->4gluino experiments
Returns a dataloader for the training data, and tensors for validation and test
'''
def build_dataloaders(config, pretrain=False):
    file_path = config["data_dir"] + "/data_4_gluino.csv"

    with FileLock(file_path):
        df = pd.read_csv(file_path, header=None)

    data_size = df.shape[0]

    n_training = config["n_training"]
    n_validation = config["n_validation"]
    n_test = config["n_test"]

    assert(n_training + n_validation + n_test <= data_size)
    
    # Only keep data we will use
    df = df[:(n_training + n_validation + n_test)]

    # Preprocessing of continuous data
    data_cont = torch.Tensor(df.values[:,:16])
    # Reshape into 4-vectors
    data_cont = data_cont.reshape(data_cont.shape[0], 4, 4)

    # Map to massless
    data_cont = to_angle_4_body(data_cont)

    # Discrete data
    data_disc = torch.IntTensor(df.values[:,16:27])
        
    # Combine and build loader    
    training_combined = torch.utils.data.TensorDataset(data_cont[ : n_training ], data_disc[ : n_training ])
    training_loader   = torch.utils.data.DataLoader(dataset=training_combined  , batch_size=config["batch_size"], shuffle=True)

    validation_combined = torch.utils.data.TensorDataset(data_cont[ n_training : n_training + n_validation ], data_disc[ n_training : n_training + n_validation ])
    validation_loader = torch.utils.data.DataLoader(dataset=validation_combined, batch_size=config["batch_size"])

    test_combined = torch.utils.data.TensorDataset(data_cont[ n_training + n_validation : n_training + n_validation + n_test ], data_disc[ n_training + n_validation : n_training + n_validation + n_test ])
    test_loader = torch.utils.data.DataLoader(dataset=test_combined, batch_size=config["batch_size"])
    
    return training_loader, validation_loader, test_loader

'''
Construct data for mixed gg->2gluino and gg->2gluino experiments
Returns a dataloader for the training data, and tensors for validation and test
'''
def build_dataloaders_mixed(config):
    file_path_4_gluino = config["data_dir"] + "/data_4_gluino.csv"
    file_path_2_gluino = config["data_dir"] + "/data_2_gluino.csv"

    with FileLock(file_path_4_gluino):
        df_4_gluino = pd.read_csv(file_path_4_gluino, header=None)
    with FileLock(file_path_2_gluino):
        df_2_gluino = pd.read_csv(file_path_2_gluino, header=None)

    data_size_4_gluino = df_4_gluino.shape[0]
    data_size_2_gluino = df_2_gluino.shape[0]

    n_training_4_gluino = int(config["n_training"]*config["four_gluino_frac"])
    n_validation_4_gluino = int(config["n_validation"]*config["four_gluino_frac"])
    n_test_4_gluino = int(config["n_test"]*config["four_gluino_frac"])

    n_training_2_gluino = int(config["n_training"]*(1 - config["four_gluino_frac"]))
    n_validation_2_gluino = int(config["n_validation"]*(1 - config["four_gluino_frac"]))
    n_test_2_gluino = int(config["n_test"]*(1 - config["four_gluino_frac"]))

    assert(n_training_4_gluino + n_validation_4_gluino + n_test_4_gluino <= data_size_4_gluino)
    assert(n_training_2_gluino + n_validation_2_gluino + n_test_2_gluino <= data_size_2_gluino)

    n_training = n_training_2_gluino + n_training_4_gluino
    n_validation = n_validation_2_gluino + n_validation_4_gluino
    n_test = n_test_2_gluino + n_test_4_gluino

    n_4_gluino = n_training_4_gluino + n_validation_4_gluino + n_test_4_gluino
    n_2_gluino = n_training_2_gluino + n_validation_2_gluino + n_test_2_gluino
    
    # Only keep data we will use
    df_4_gluino = df_4_gluino[:n_4_gluino]
    df_2_gluino = df_2_gluino[:n_2_gluino]

    # We only use continuous data for the mixed experiments
    data_4_gluino = torch.Tensor(df_4_gluino.values[:,:16])
    data_2_gluino = torch.Tensor(df_2_gluino.values[:,:8])
    data_2_gluino = torch.cat((data_2_gluino, torch.zeros(data_2_gluino.shape[0], 8)), dim=-1)

    # Reshape 
    data_4_gluino = data_4_gluino.reshape(data_4_gluino.shape[0], 4, 4)
    data_2_gluino = data_2_gluino.reshape(data_2_gluino.shape[0], 4, 4)

    # Map to massless
    data_4_gluino = to_angle_4_body(data_4_gluino)
    data_2_gluino = to_angle_2_body(data_2_gluino)

    # Concatenate data
    data_combine = torch.cat((data_4_gluino, data_2_gluino))

    # Shuffle data
    data_combine = data_combine[torch.randperm(data_combine.shape[0])]

    # Set up data loader
    training_data = data_combine[ : n_training ]
    training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=config["batch_size"], shuffle=True)

    validation_data = data_combine[ n_training : n_training + n_validation ]
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=config["batch_size"], shuffle=True)

    test_data = data_combine[ n_training + n_validation : n_training + n_validation + n_test ]
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=True)
    
    return training_loader, validation_loader, test_loader

'''
Generate the discrete dictionary
'''
def generate_discrete_dictionary(data_path, dict_path, n_train):
    with FileLock(data_path):
        df = pd.read_csv(data_path, header=None)

    encoder = GluinoCategoricalEncoder()

    permute_classes_discrete = np.array([[2,7],[3,8],[4,9],[5,10]])
    permuter = IteratedPermutation(None, permute_classes_discrete)

    category_dict = {}
    total_samples = 0
    with torch.no_grad():
        data_disc = torch.IntTensor(df.values[:n_train, 16:27])

        while True:
            _, permuted_data_disc, _ = permuter.forward(None, data_disc)

            # Encode
            encoded_data_disc = encoder.encode(permuted_data_disc)

            # Encode to a single number
            encoded_num = encoded_data_disc[:,0] + encoded_data_disc[:,1]*64

            # Fill dictionary
            for category in encoded_num:
                total_samples += 1
                if category.item() in category_dict:
                    category_dict[category.item()] += 1
                else:
                    category_dict[category.item()] = 1
            
            # Advance permuter
            if not permuter.next():
                break

    for key in category_dict:
        category_dict[key] = category_dict[key]

    category_dict["tot_samples"] = total_samples

    file_out = open(dict_path, 'w+')
    yaml.dump(category_dict, file_out, allow_unicode=True)
