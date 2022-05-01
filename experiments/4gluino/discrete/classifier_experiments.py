import torch
import torch.optim as optim
from torch import nn
import os
import argparse
import yaml
import tqdm
from tqdm import trange

from surflows.gluino.gluino_auxiliary import build_dataloaders
from surflows.gluino.gluino_models import DiscreteClassifier

def compute_loss_over_dataloader(model, dataloader, device):
    loss = 0
    data_size = 0
    cross_entropy_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_cont, batch_disc in dataloader:
            output, categories = model.forward_train(batch_cont.to(device), batch_disc.to(device))
            loss_now = cross_entropy_loss(output, categories)

            loss = loss*data_size + loss_now*batch_cont.shape[0]
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

    model = DiscreteClassifier(hyperparams)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams["classifier_lr"])

    # Scheduling with reduction on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams["lr_decay"], patience=hyperparams["lr_decay_patience"], verbose=True)

    # Loss
    cross_entropy_loss = nn.CrossEntropyLoss()

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


            output, categories = model.forward_train(train_batch_cont.to(device), train_batch_disc.to(device))

            loss = cross_entropy_loss(output, categories)

            loss.backward()
            optimizer.step()

            # ---------------- Validation -----------------
            if i == hyperparams["validation_interval"]-1:
                with torch.no_grad():
                    validation_counter += 1
                    batch_counter = 0

                    # Compute validation loss
                    validation_loss = compute_loss_over_dataloader(model, val_loader, device)

                    # Compute test loss and save best model
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        test_loss = compute_loss_over_dataloader(model, test_loader, device)
                        torch.save(model.state_dict(), "results/model_classifier.pt")
                    
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
parser.add_argument('--classifier_size', type=int)
parser.add_argument('--classifier_layers', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--gpu_device', type=int)
args = parser.parse_args()

hyperparams["permutation"] = "stochastic"

if args.learning_rate is not None:
    hyperparams["classifier_lr"] = args.learning_rate
if args.classifier_size is not None:
    hyperparams["classifier_size"] = args.classifier_size
if args.classifier_layers is not None:
    hyperparams["classifier_layers"] = args.classifier_layers

train(args, hyperparams)
