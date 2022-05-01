import torch
import torch.optim as optim
import wandb
import os
import argparse
import yaml
import numpy as np
import copy
import sys
import tqdm
from tqdm import trange

from surflows.darkmachines.darkmachines_model import DarkMachinesMixtureModel, DarkMachinesDequantizationModel, DarkMachinesClassifierModel
from surflows.darkmachines.darkmachines_auxiliary import DarkMachinesData

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d

with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)
    
# Add data dir
hyperparams["data_dir"] = os.getcwd()

parser = argparse.ArgumentParser(description='Anomaly detection')
parser.add_argument('--perm', required=True, choices=['stochastic', 'ordered'])
parser.add_argument('--model', required=True, choices=['mixture', 'dequantization', 'classifier'])
parser.add_argument('--num_objects', required=True, type=int)
parser.add_argument('--gpu_device', type=int)
args = parser.parse_args()

hyperparams['permutation'] = args.perm
hyperparams['num_objects'] = args.num_objects

if args.gpu_device is None:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_device)

model_name = args.model + "_" + hyperparams['permutation'] + "_" + str(hyperparams["num_objects"]) + "_objects"

# Directory for results
if not os.path.exists('results/' + model_name):
    os.mkdir('results/' + model_name)

test_scores_file = open('results/' + model_name + "/test_scores.dat", "w")

for channel in ['1', '2a', '2b', '3']:
    hyperparams['channel'] = channel

    # ---------------------------------- Load the data ----------------------------------
    data_obj = DarkMachinesData(hyperparams)
    train_loader, val_loader, test_loaders, secret_loader = data_obj.convert_channel_to_dataloaders()

    # ---------------------------------- Set up the model ----------------------------------
    if args.model == 'mixture':
        model = DarkMachinesMixtureModel(hyperparams)
    elif args.model == 'dequantization':
        model = DarkMachinesDequantizationModel(hyperparams)
    elif args.model == 'classifier':
        model = DarkMachinesClassifierModel(hyperparams)
    model.to(device)

    # ---------------------------------- Train the model ----------------------------------   
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Scheduling with reduction on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams["lr_decay"], patience=hyperparams["lr_decay_patience"], verbose=True)
        
    # Train loop
    batch_generator = iter(train_loader)
    best_validation_loss = float("inf")
    validation_counter = 0
    best_state_dict = None
    stop = False 
    while True:
        if stop:
            break

        tqdm_range = trange(hyperparams["validation_interval"])
        for i in tqdm_range:
            # Catch dataloader exceptions when hitting end of dataset
            try:
                train_batch_cont, train_batch_disc = next(batch_generator)
            except StopIteration:
                # Restart the generator
                batch_generator = iter(train_loader)
                train_batch_cont, train_batch_disc = next(batch_generator)

            # SGD
            optimizer.zero_grad()
            if args.model == 'classifier':
                loss = -model.log_prob_flow(train_batch_cont.to(device), train_batch_disc.to(device)).mean()
            else:
                loss = -model.log_prob(train_batch_cont.to(device), train_batch_disc.to(device)).mean()
            loss.backward()
            optimizer.step()

            # ---------------- Validation -----------------
            if i == hyperparams["validation_interval"]-1:
                with torch.no_grad():
                    validation_counter += 1

                    # Compute validation loss
                    validation_loss = 0
                    val_num_samples = 0
                    for val_batch_cont, val_batch_disc in val_loader:
                        # Filter out infs due to nonexisting categorical prob
                        if args.model == 'classifier':
                            val_log_probs = model.log_prob_flow(val_batch_cont.to(device), val_batch_disc.to(device))
                        else:
                            val_log_probs = model.log_prob(val_batch_cont.to(device), val_batch_disc.to(device))
                        val_log_probs = val_log_probs[~torch.isinf(val_log_probs)]
                        validation_loss -= val_log_probs.sum()
                        val_num_samples += val_log_probs.shape[0]
                    validation_loss /= val_num_samples
                    tqdm_range.set_description("Validation loss = " + str(validation_loss.item()))

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_state_dict = copy.deepcopy(model.state_dict())

                    # Update lr
                    scheduler.step(validation_loss)
                    
                    # Break when lr has been reduced by 2 orders of magnitude
                    if optimizer.param_groups[0]['lr'] < hyperparams["learning_rate"]*1e-2 or validation_counter == 2000:
                        stop = True
            else:    
                tqdm_range.set_description("Batch loss = " + str(loss.item()))
    # ---------------------------------- Train the classifier ----------------------------------
    if args.model == 'classifier':
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["classifier_lr"])

        # Scheduling with reduction on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams["lr_decay"], patience=hyperparams["lr_decay_patience"], verbose=True)

        batch_generator = iter(train_loader)
        best_validation_loss = float("inf")
        validation_counter = 0
        best_state_dict_classifier = None

        while True:
            tqdm_range = trange(hyperparams["validation_interval"])
            for i in tqdm_range:
                # Catch dataloader exceptions when hitting end of dataset
                try:
                    train_batch_cont, train_batch_disc = next(batch_generator)
                except StopIteration:
                    # Restart the generator
                    batch_generator = iter(train_loader)
                    train_batch_cont, train_batch_disc = next(batch_generator)

                # SGD
                optimizer.zero_grad()
                loss = -model.log_prob_classifier(train_batch_cont.to(device), train_batch_disc.to(device)).mean()
                loss.backward()
                optimizer.step()

                # ---------------- Validation -----------------
                if i == hyperparams["validation_interval"]-1:
                    with torch.no_grad():
                        validation_counter += 1

                        # Compute validation loss
                        validation_loss = 0
                        val_num_samples = 0
                        for val_batch_cont, val_batch_disc in val_loader:
                            val_log_probs = model.log_prob_classifier(val_batch_cont.to(device), val_batch_disc.to(device))
                            validation_loss -= val_log_probs.sum()
                            val_num_samples += val_log_probs.shape[0]
                        validation_loss /= val_num_samples

                        tqdm_range.set_description("Validation loss = " + str(validation_loss.item()))

                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            best_state_dict_classifier = copy.deepcopy(model.classifier_.state_dict())

                        # Update lr
                        scheduler.step(validation_loss)
                        
                        # Break when lr has been reduced by 2 orders of magnitude
                        if optimizer.param_groups[0]['lr'] < hyperparams["learning_rate"]*1e-2 or validation_counter == 2000:
                            break
                else:    
                    tqdm_range.set_description("Batch loss = " + str(loss.item()))

    # ---------------------------------- Evaluation of the best model ----------------------------------
    model.load_state_dict(best_state_dict)
    if args.model == 'classifier':
        model.classifier_.load_state_dict(best_state_dict_classifier)
    model.eval()

    with torch.no_grad():
        # Test channel scores
        for loader, channel_name in test_loaders:
            log_probs = []
            labels = []
            for batch_cont, batch_disc, batch_labels in loader:
                batch_log_probs = model.log_prob(batch_cont.to(device), batch_disc.to(device))
                log_probs.append(batch_log_probs.detach().cpu().numpy())
                labels.append(batch_labels.detach().cpu().numpy())
            log_probs = np.concatenate(log_probs)
            labels = np.concatenate(labels)
            
            log_prob_max = np.max(log_probs)
            log_prob_min = np.min(log_probs[log_probs != -np.inf])

            scores = (log_prob_max - log_probs)/(log_prob_max - log_prob_min)
            # inf scores correspond with categorical configurations that do not exist in the training data
            # They are thus maximally anomalous
            scores[scores == np.inf] = 1
            
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = roc_auc_score(y_true=labels, y_score=scores)
            roc_interp = interp1d(fpr, tpr)

            channel_name = channel_name.split(".")[0]

            test_scores_file.write(channel_name + " " + model_name + " " + hyperparams['channel'] + " " + str(roc_auc) + " " + str(roc_interp(1e-2)) + " " + str(roc_interp(1e-3)) + " " + str(roc_interp(1e-4)) + "\n")

        # Secret dataset
        log_probs = []
        for batch_cont, batch_disc in secret_loader:
            batch_log_probs = model.log_prob(batch_cont.to(device), batch_disc.to(device))
            log_probs.append(batch_log_probs.detach().cpu().numpy())
        log_probs = np.concatenate(log_probs)

        log_prob_max = np.max(log_probs)
        log_prob_min = np.min(log_probs[log_probs != -np.inf])
        scores = (log_prob_max - log_probs)/(log_prob_max - log_prob_min)
        scores[scores == np.inf] = 1

        np.savetxt('results/' + model_name + "/" + hyperparams['channel'] + "_scores.dat", scores, fmt='%1.4f')



