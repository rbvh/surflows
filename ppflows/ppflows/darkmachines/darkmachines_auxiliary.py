# Parts of the code below is takes from the DarkMachines collaboration

import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import math
from ppflows.file_lock import FileLock
from ppflows.permuters import IteratedPermutation
import yaml

# Convert data to torch tensors
# Also convert energy to mass, adding a 0 if there is no jet
# Output format is 
# [MET, METphi, pT1, eta1, phi1, E1, pT2, m2, eta2, phi2, E2, .....]
# [obj1, obj2, ....]
# Energies are not stored for objects that are not jets, as they contain no additional information 
# All energy-scaling observables are log-transformed
# Categorical encoding for objects is 
# 0 - empty
# 1 - jet (j)
# 2 - b-jet (b)
# 3 - photon (g)
# 4 - positron (e+)
# 5 - electro (e-)
# 6 - antimuon (m+)
# 7 - muon (m-)

class DarkMachinesData(nn.Module):
    def __init__(self, hyperparams):
        super(DarkMachinesData, self).__init__()

        self.categorical_map_ = {
            "j": 3,
            "b": 4,
            "g": 5,
            "e+": 1,
            "e-": 2,
            "m+": 6,
            "mu+": 6,
            "m-": 7,
            "mu-": 7,
        }

        self.num_objects_           = hyperparams["num_objects"]
        self.channel_               = hyperparams["channel"]
        self.file_dir_              = hyperparams["data_dir"]
        self.file_dir_train_        = hyperparams["data_dir"] + "/training_files/chan" + hyperparams["channel"]
        self.file_dir_secret_       = hyperparams["data_dir"] + "/secret_data/chan"    + hyperparams["channel"]
        self.validation_fraction_   = hyperparams["validation_fraction"]
        self.test_fraction_         = hyperparams["test_fraction"]
        self.batch_size_            = hyperparams["batch_size"]

        self.register_buffer("means_", torch.zeros(2 + 4*self.num_objects_))
        self.register_buffer("stds_",  torch.zeros(2 + 4*self.num_objects_))

    def convert_to_tensor(self, file_name, drop_energy = False):
        with FileLock(file_name):
            num_lines = sum(1 for line in open(file_name))

            continuous_data = torch.zeros(num_lines, 2 + 4*self.num_objects_)*float('nan')
            discrete_data = torch.zeros(num_lines, self.num_objects_).long()

            with open(file_name, 'r') as file:
                for i, line in enumerate(file.readlines()):
                    line = line.replace(';', ',')
                    line = line.rstrip(',\n')
                    line = line.split(',')

                    continuous_data[i][0] = math.log(float(line[3]))
                    continuous_data[i][1] = float(line[4])

                    for j in range(0, min(int((len(line) - 5)/5), self.num_objects_)):
                        discrete_data[i][j] = self.categorical_map_[line[5*j+5]]

                        continuous_data[i][2+4*j] = math.log(float(line[5*j + 7]))
                        if not drop_energy or (discrete_data[i][j] == 1 or discrete_data[i][j] == 2):
                            continuous_data[i][3+4*j] = math.log(float(line[5*j+6]))
                        continuous_data[i][4+4*j] = float(line[5*j + 8])
                        continuous_data[i][5+4*j] = float(line[5*j + 9])

        return continuous_data, discrete_data

    def convert_channel_to_dataloaders(self):
        # ----------------------------------- Training data -----------------------------------
        for file_name in os.scandir(self.file_dir_train_):
            if "background" in file_name.name:
                training_data_continuous, training_data_discrete = self.convert_to_tensor(file_name.path)
                break

        # Compute the normalization
        # Have to do this sequentially because of the presence of nans
        for i in range(training_data_continuous.shape[1]):
            col_without_nans = training_data_continuous[:,i]
            col_without_nans = col_without_nans[~col_without_nans.isnan()]
            self.means_[i] = col_without_nans.mean()
            self.stds_[i]  = col_without_nans.std()

        # Normalize
        training_data_continuous = (training_data_continuous - self.means_)/self.stds_
        n_val = int(self.validation_fraction_*training_data_continuous.shape[0])
        n_test = int(self.test_fraction_*training_data_continuous.shape[0])

        # Split the data
        # Make sure the test data is always at the first fraction, ensuring it is consistent if n_val changes
        # Add on labels
        training_data_combined = torch.utils.data.TensorDataset(training_data_continuous[n_val+n_test:], training_data_discrete[n_val+n_test:])
        validation_data_combined = torch.utils.data.TensorDataset(training_data_continuous[n_val:n_val+n_test], training_data_discrete[n_val:n_val+n_test])
        test_data_combined = torch.utils.data.TensorDataset(training_data_continuous[:n_test], training_data_discrete[:n_test], torch.zeros(n_test))
        
        # Construct dataloaders
        training_loader = torch.utils.data.DataLoader(dataset=training_data_combined, batch_size=self.batch_size_, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_data_combined, batch_size=self.batch_size_)

        # ----------------------------------- Signal data -----------------------------------
        test_loaders = []
        for file_name in os.scandir(self.file_dir_train_):
            if "background" not in file_name.name:
                data_continuous, data_discrete = self.convert_to_tensor(file_name.path)
                # Normalize
                data_continuous = (data_continuous - self.means_)/self.stds_
        
                # Store
                signal_data_combined = torch.utils.data.TensorDataset(data_continuous, data_discrete, torch.ones(data_continuous.shape[0]))

                combined_data = torch.utils.data.ConcatDataset((test_data_combined, signal_data_combined))
                test_loaders.append([torch.utils.data.DataLoader(dataset=combined_data, batch_size=self.batch_size_), file_name.name])

        # ----------------------------------- Secret data -----------------------------------
        for file_name in os.scandir(self.file_dir_secret_):
            secret_data_continuous, secret_data_discrete = self.convert_to_tensor(file_name.path)
            # Normalize
            secret_data_continuous = (secret_data_continuous - self.means_)/self.stds_

        secret_data_combined = torch.utils.data.TensorDataset(secret_data_continuous, secret_data_discrete)
        secret_loader = torch.utils.data.DataLoader(dataset=secret_data_combined, batch_size=self.batch_size_)

        return training_loader, validation_loader, test_loaders, secret_loader

    # Undo normalization and log scaling 
    def unnormalize(self, inputs_continuous):
        outputs_continuous = inputs_continuous*self.stds_ + self.means_
        outputs_continuous[:,0] = torch.exp(outputs_continuous[:,0])
        for i in range(int((inputs_continuous.shape[0]-2)/4)):
            outputs_continuous[:, 2+4*i] = torch.exp(outputs_continuous[:, 2+4*i])
            outputs_continuous[:, 5+4*i] = torch.exp(outputs_continuous[:, 5+4*i])

        return outputs_continuous

    def generate_dictionary(self, do_categorical = False):
        with torch.no_grad():
            # Get the training data
            for file_name in os.scandir(self.file_dir_train_):
                if "background" in file_name.name:
                    _, data_disc = self.convert_to_tensor(file_name.path)
                    break

            # Compute the factors required for the encoding of categorical labels 
            num_categories_per_dim = torch.ones(self.num_objects_)*8
            category_factors = torch.ones(num_categories_per_dim.shape[0]).long()
            for i in range(1, num_categories_per_dim.shape[0]):
                category_factors[i] = category_factors[i-1]*num_categories_per_dim[i-1]

            # Compute the factors required for the encoding of categorical labels     
            dropout_factors = torch.ones(num_categories_per_dim.shape[0]).long()
            for i in range(1, num_categories_per_dim.shape[0]):
                dropout_factors[i] = dropout_factors[i-1]*2
            
            # ---------------------------- Generate the ordered dictionaries ----------------------------
            category_dict = {}
            dropout_dict = {}
            total_samples = 0

            dropout_encoded = torch.sum(dropout_factors*(data_disc != 0), dim=-1).squeeze().numpy()
            # Fill dropout dictionary
            for dropout in dropout_encoded:
                total_samples += 1
                if dropout.item() in dropout_dict:
                    dropout_dict[dropout.item()] += 1
                else:
                    dropout_dict[dropout.item()] = 1
            dropout_dict["tot_samples"] = total_samples

            file_name = self.file_dir_ + '/dicts/dropout_dict_chan_' + self.channel_ + "_ordered_num_objects_" + str(self.num_objects_) + ".yaml"
            file_out = open(file_name, 'w+')
            yaml.dump(dropout_dict, file_out, allow_unicode=True)

            if do_categorical:
                # Fill categorical dictionary
                categories_encoded = torch.sum(category_factors*data_disc, dim=-1).squeeze().numpy()
                for category in categories_encoded:
                    if category.item() in category_dict:
                        category_dict[category.item()] += 1
                    else:
                        category_dict[category.item()] = 1
                category_dict["tot_samples"] = total_samples

                file_name = self.file_dir_ + '/dicts/category_dict_chan_' + self.channel_ + "_ordered_num_objects_" + str(self.num_objects_) + ".yaml"
                file_out = open(file_name, 'w+')
                yaml.dump(category_dict, file_out, allow_unicode=True)

            # ---------------------------- Generate the stochastic categorical dictionary ----------------------------

            if do_categorical:
                # Set up the permutation model
                permute_classes_discrete = []
                for i in range(self.num_objects_):
                    permute_classes_discrete.append(np.array([i]))
                permute_classes_discrete = np.array(permute_classes_discrete)
                permuter = IteratedPermutation(None, permute_classes_discrete)

                # Set up a tensor with all the classes
                categories_in_dict = torch.empty(len(category_dict)-1).long()
                counts_in_dict = torch.empty(len(category_dict)-1).long()
                for i, key in enumerate(category_dict):
                    if key != 'tot_samples':
                        categories_in_dict[i] = key
                        counts_in_dict[i] = category_dict[key]

                # Decode the classes
                categories_decoded = torch.ones((categories_in_dict.shape[0], self.num_objects_), dtype=torch.long)
                for i in range(self.num_objects_-1, -1, -1):
                    categories_decoded[:,i] = (categories_in_dict - categories_in_dict % category_factors[i])/category_factors[i]
                    categories_in_dict -= categories_decoded[:,i]*category_factors[i]

                # Now do permutations
                category_dict = {}
                
                while True:
                    categories_permuted = permuter.forward(None, categories_decoded)[1]
                    categories_encoded = torch.sum(category_factors*categories_permuted, dim=-1).long().squeeze().numpy()

                    # Fill in categorical dictionary
                    for i, category in enumerate(categories_encoded):
                        if category.item() in category_dict:
                            category_dict[category.item()] += counts_in_dict[i].item()
                        else:
                            category_dict[category.item()] = counts_in_dict[i].item()
                                        
                    # Advance permuter
                    if not permuter.next():
                        break

                category_dict["tot_samples"] = permuter.permutations_.shape[0] * total_samples

                # Output the stochastic dictionaries
                file_name = self.file_dir_ + '/dicts/category_dict_chan_' + self.channel_ + "_stochastic_num_objects_" + str(self.num_objects_) + ".yaml"
                file_out = open(file_name, 'w+')
                yaml.dump(category_dict, file_out, allow_unicode=True)


            # ---------------------------- Generate the stochastic dropout dictionary ----------------------------
            # Set up the permutation model
            permute_classes_discrete = []
            for i in range(self.num_objects_):
                permute_classes_discrete.append(np.array([i]))
            permute_classes_discrete = np.array(permute_classes_discrete)
            permuter = IteratedPermutation(None, permute_classes_discrete)

            # Set up a tensor with all the dropout configurations
            dropouts_in_dict = torch.empty(len(dropout_dict)-1).long()
            counts_in_dict = torch.empty(len(dropout_dict)-1).long()
            for i, key in enumerate(dropout_dict):
                if key != 'tot_samples':
                    dropouts_in_dict[i] = key
                    counts_in_dict[i] = dropout_dict[key]

            # Decode the dropout configurations
            dropouts_decoded = torch.ones((dropouts_in_dict.shape[0], self.num_objects_), dtype=torch.long)
            for i in range(self.num_objects_-1, -1, -1):
                dropouts_decoded[:,i] = (dropouts_in_dict - dropouts_in_dict % dropout_factors[i])/dropout_factors[i]
                dropouts_in_dict -= dropouts_decoded[:,i]*dropout_factors[i]

            dropout_dict = {}

            while True:
                if permuter.counter_ % 10000 == 0:
                    print(permuter.counter_)
                dropouts_permuted = permuter.forward(None, dropouts_decoded)[1]
                dropout_encoded = torch.sum(dropout_factors*dropouts_permuted, dim=-1).long().squeeze().numpy()  

                # Fill dropout dictionary
                for i, dropout in enumerate(dropout_encoded):
                    if dropout.item() in dropout_dict:
                        dropout_dict[dropout.item()] += counts_in_dict[i].item()
                    else:
                        dropout_dict[dropout.item()] = counts_in_dict[i].item()

                # Advance permuter
                if not permuter.next():
                    break

            dropout_dict["tot_samples"] = permuter.permutations_.shape[0] * total_samples

            file_name = self.file_dir_ + '/dicts/dropout_dict_chan_' + self.channel_ + "_stochastic_num_objects_" + str(self.num_objects_) + ".yaml"
            file_out = open(file_name, 'w+')
            yaml.dump(dropout_dict, file_out, allow_unicode=True)