#!/usr/bin/python3

import torch
import torch.optim as optim
import os
import argparse
import yaml
import tqdm
from tqdm import trange

from ppflows.gluino.gluino_auxiliary import build_dataloaders
from ppflows.gluino.gluino_models import GluinoModel
from ppflows.utils import EarlyStopper

def compute_loss_over_dataloader(model, dataloader, device):
    loss = 0
    data_size = 0
    with torch.no_grad():
        for batch_cont, _ in dataloader:
            loss_now = -model.log_prob(batch_cont.to(device)).mean()

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

    model = GluinoModel(hyperparams)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Scheduling with reduction on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams["lr_decay"], patience=hyperparams["lr_decay_patience"])

    batch_generator = iter(train_loader)
    best_validation_loss = float("inf")
    batch_counter = 0
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
                train_batch_cont, _ = next(batch_generator)
            except StopIteration:
                # Restart the generator
                batch_generator = iter(train_loader)
                train_batch_cont, _ = next(batch_generator)

            # SGD
            optimizer.zero_grad()
            loss = -model.log_prob(train_batch_cont.to(device)).mean()    
            loss.backward()
            optimizer.step()

            if i == hyperparams["validation_interval"]-1:
                # ---------------- Validation -----------------
                validation_counter += 1

                # Compute validation loss
                validation_loss = compute_loss_over_dataloader(model, val_loader, device)
                tqdm_range.set_description("Validation loss = " + str(validation_loss))

                # Compute test loss and save best model
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    test_loss = compute_loss_over_dataloader(model, test_loader, device)
                    torch.save(model.state_dict(), "results/model_" + hyperparams["permutation"] + "_" + args.ntrain + ".pt")
                    
                # Update lr
                scheduler.step(validation_loss)

                # Break when lr has been reduced by 3 orders of magnitude or after 10k validations
                if optimizer.param_groups[0]['lr'] < hyperparams["learning_rate"]*1e-3 or validation_counter == 10000:
                    stop = True
            else:    
                tqdm_range.set_description("Batch loss = " + str(loss.item()))

    print("Test loss = " + str(test_loss))
                        
with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)

# Add data dir
hyperparams["data_dir"] = os.getcwd() + "/.."

parser = argparse.ArgumentParser(description='Permutation experiments')
parser.add_argument('--ntrain', required=True, choices=['50k', '100k', '200k', '500k', '750k', '1M'], help='n_train')
parser.add_argument('--gpu_device', type=int)
args = parser.parse_args()

if args.ntrain == "50k":
    hyperparams["n_training"] = 50000
elif args.ntrain == "100k":
    hyperparams["n_training"] = 100000
elif args.ntrain == "200k":
    hyperparams["n_training"] = 200000
elif args.ntrain == "500k":
    hyperparams["n_training"] = 500000
elif args.ntrain == "750k":
    hyperparams["n_training"] = 750000
elif args.ntrain == "1M":
    hyperparams["n_training"] = 1000000

# All types of permutations
hyperparams["permutation"] = "none"
train(args, hyperparams)

hyperparams["permutation"] = "stochastic"
train(args, hyperparams)

hyperparams["permutation"] = "ordered"
train(args, hyperparams)