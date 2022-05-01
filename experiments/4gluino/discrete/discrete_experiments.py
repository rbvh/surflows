#!/usr/bin/python3

import torch
import torch.optim as optim
import os
import argparse
import yaml
import tqdm
from tqdm import trange

from ppflows.gluino.gluino_auxiliary import build_dataloaders
from ppflows.gluino.gluino_models import GluinoModel, MixtureGluinoModel

def compute_balanced_mixture_loss(model, batch_cont, batch_disc):
    # Encode discrete features
    code = model.encoder_.encode(batch_disc)
    code = code[:,0] + code[:,1]*64

    # Get the counts of all codes, as well as their locations
    _, inverse, counts = torch.unique(code, return_inverse=True, return_counts=True)

    # Set up weights that cancel the categorical probability and normalize them
    weights_per_category = code.shape[0]/counts
    weights_per_category = code.shape[0] * weights_per_category / torch.sum(weights_per_category)
    
    # Find weights per element of the batch
    weights = weights_per_category[inverse]

    # Compute loss
    return -( model.log_prob(batch_cont, batch_disc)*weights ).mean()

def compute_loss_over_dataloader(model, dataloader, device):
    loss = 0
    data_size = 0
    with torch.no_grad():
        for batch_cont, batch_disc in dataloader:
            loss_now = -model.log_prob(batch_cont.to(device), batch_disc.to(device)).mean()

            loss = loss*data_size + loss_now.item()*batch_cont.shape[0]
            data_size += batch_cont.shape[0]
            loss /= data_size

    return loss

def train(args, hyperparams):
    if args.gpu_device is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)

    train_loader, val_loader, test_loader = build_dataloaders(hyperparams)

    if "mixture" in hyperparams["discrete_mode"]:
        model = MixtureGluinoModel(hyperparams)
    else:
        model = GluinoModel(hyperparams)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Scheduling with reduction on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams["lr_decay"], patience=hyperparams["lr_decay_patience"], verbose=True)

    batch_generator = iter(train_loader)
    best_validation_loss = float("inf")
    validation_counter = 0
    test_loss = 0
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
            if hyperparams["discrete_mode"] == "mixture_balanced":
                loss = compute_balanced_mixture_loss(model, train_batch_cont.to(device), train_batch_disc.to(device))
            else:
                loss = -model.log_prob(train_batch_cont.to(device), train_batch_disc.to(device)).mean()

            loss.backward()
            optimizer.step()

            # ---------------- Validation -----------------
            if i == hyperparams["validation_interval"]-1:
                with torch.no_grad():
                    validation_counter += 1

                    # Compute validation loss
                    validation_loss = compute_loss_over_dataloader(model, val_loader, device)
                    tqdm_range.set_description("Validation loss = " + str(validation_loss))

                    # Compute test loss and save best model
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        test_loss = compute_loss_over_dataloader(model, test_loader, device)
                        torch.save(model.state_dict(), "results/model_" + hyperparams["discrete_mode"] + "_" + hyperparams["permutation"] + ".pt")
                        
                    # Update lr
                    scheduler.step(validation_loss)
                    
                    # Break when lr has been reduced by 3 orders of magnitude
                    if optimizer.param_groups[0]['lr'] < hyperparams["learning_rate"]*1e-3 or validation_counter == 10000:
                        stop = True
            else:
                tqdm_range.set_description("Batch loss = " + str(loss.item()))

                
with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)
    
# Add data dir
hyperparams["data_dir"] = os.getcwd() + "/.."

parser = argparse.ArgumentParser(description='Discrete experiments')
parser.add_argument('--mode', required=True, choices=['uniform_dequantization', 'flow_dequantization', 'uniform_argmax', 'flow_argmax', 'mixture_likelihood', 'mixture_balanced'], help='Discrete mode')
parser.add_argument('--perm', required=True, choices=['stochastic', 'ordered'])
parser.add_argument('--gpu_device', type=int)
args = parser.parse_args()

hyperparams["discrete_mode"] = args.mode
hyperparams["permutation"] = args.perm

# Scale down the model for argmax
if "argmax" in args.mode:
    hyperparams["n_made_units_per_dim"] = 4
    hyperparams["n_RQS_knots"] = 8

train(args, hyperparams)