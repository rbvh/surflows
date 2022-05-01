#!/usr/bin/python3

import torch
import torch.optim as optim
import os
import argparse
import yaml
import tqdm
from tqdm import trange

from surflows.gluino.gluino_auxiliary import build_dataloaders_mixed
from surflows.gluino.gluino_models import DropoutGluinoModel

def compute_loss_components(model, data):
    log_probs_conditional = model.log_prob_conditional(data)
    log_prob_dropout = log_probs_conditional[torch.any(data == -1., dim=1)]
    log_prob_no_dropout = log_probs_conditional[~torch.any(data == -1., dim=1)]

    return log_prob_dropout, log_prob_no_dropout

def compute_loss(model, data, norm_dropout):
    log_prob_dropout, log_prob_no_dropout = compute_loss_components(model, data)

    return -norm_dropout*log_prob_dropout.mean() - (1. - norm_dropout)*log_prob_no_dropout.mean()

def compute_losses_over_dataloader(model, dataloader, norm_dropout, device):
    loss_mean = 0
    loss_num = 0
    log_prob_dropout_mean = 0
    log_prob_dropout_num = 0
    log_prob_no_dropout_mean = 0
    log_prob_no_dropout_num = 0
    with torch.no_grad():
        for batch in dataloader:
            log_prob_dropout, log_prob_no_dropout = compute_loss_components(model, batch.to(device))

            log_prob_dropout_mean = log_prob_dropout_mean*log_prob_dropout_num + torch.sum(log_prob_dropout)
            log_prob_dropout_num += log_prob_dropout.shape[0]
            log_prob_dropout_mean /= log_prob_dropout_num
            
            log_prob_no_dropout_mean = log_prob_no_dropout_mean*log_prob_no_dropout_num + torch.sum(log_prob_no_dropout)
            log_prob_no_dropout_num += log_prob_no_dropout.shape[0]
            log_prob_no_dropout_mean /= log_prob_no_dropout_num

    loss = -norm_dropout*log_prob_dropout_mean - (1. - norm_dropout)*log_prob_no_dropout_mean
    return log_prob_dropout_mean, log_prob_no_dropout_mean, loss

def train(args, hyperparams):
    if args.gpu_device is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)

    train_loader, val_loader, test_loader = build_dataloaders_mixed(hyperparams)

    model = DropoutGluinoModel(hyperparams)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    if args.mode == 'likelihood':
        norm_dropout = 0.99995993045
    elif args.mode == 'balanced':
        norm_dropout = 0.5
    elif args.mode == 'biased':
        norm_dropout = 1e-4

    # Scheduling with reduction on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams["lr_decay"], patience=hyperparams["lr_decay_patience"])

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
                train_batch_cont = next(batch_generator)
            except StopIteration:
                # Restart the generator
                batch_generator = iter(train_loader)
                train_batch_cont = next(batch_generator)

            # SGD
            optimizer.zero_grad()
            loss = compute_loss(model, train_batch_cont.to(device), norm_dropout)
            loss.backward()
            optimizer.step()
            
            # ---------------- Validation -----------------
            if i == hyperparams["validation_interval"]-1:
                with torch.no_grad():
                    validation_counter += 1

                    # Compute validation loss
                    validation_log_prob_dropout, validation_log_prob_no_dropout, validation_loss = compute_losses_over_dataloader(model, val_loader, norm_dropout, device)
                    tqdm_range.set_description("Validation loss = " + str(validation_loss.item()))

                    # Compute test loss and save best model
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        test_log_prob_dropout, test_log_prob_no_dropout, test_loss = compute_losses_over_dataloader(model, test_loader, norm_dropout, device)
                        torch.save(model.state_dict(), "results/model_" + hyperparams["permutation"] + "_" + args.mode + ".pt")
                                
                    # Update lr
                    scheduler.step(validation_loss)

                    # Break when lr has been reduced by 3 orders of magnitude or after 10k validations
                    if optimizer.param_groups[0]['lr'] < hyperparams["learning_rate"]*1e-3 or validation_counter == 10000:
                        stop = True
            else:
                tqdm_range.set_description("Batch loss = " + str(loss.item()))

                
with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)

# Add data dir
hyperparams["data_dir"] = os.getcwd() + "/.."

parser = argparse.ArgumentParser(description='Permutation experiments')
parser.add_argument('--perm', required=True, choices=['stochastic', 'ordered'])
parser.add_argument('--mode', required=True, choices=['likelihood', 'balanced', 'biased'])
parser.add_argument('--gpu_device', type=int)
args = parser.parse_args()

hyperparams["permutation"] = args.perm

train(args, hyperparams)