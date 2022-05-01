#!/usr/bin/env python

import torch
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import yaml
import argparse
import os

from ppflows.permuters import IteratedPermutation
from ppflows.gluino.gluino_models import GluinoModel, MixtureGluinoModel, ClassifierGluinoModel
from ppflows.gluino.gluino_auxiliary import to_phase_space_4_body
from ppflows.utils import Histogram

def perp(x, y):
    z = np.zeros((x.shape[0], 3))
    z[:,0] = x[:,2]*y[:,3] - x[:,3]*y[:,2]
    z[:,1] = x[:,3]*y[:,1] - x[:,1]*y[:,3]
    z[:,2] = x[:,1]*y[:,2] - x[:,2]*y[:,1]

    # Normalize
    norm = np.expand_dims(np.sqrt(z[:,0]**2 + z[:,1]**2 + z[:,2]**2), axis=1)
    return z/norm

def dpsi(p):
    perp_1 = perp(p[:,0], p[:,1])
    perp_2 = perp(p[:,0] + p[:,1], p[:,2])

    dpsi = np.arccos(perp_1[:,0]*perp_2[:,0] + perp_1[:,1]*perp_2[:,1] + perp_1[:,2]*perp_2[:,2])

    return dpsi

n_train = 1000000

modes = ["uniform_dequantization", "flow_dequantization", "uniform_argmax", "flow_argmax", "mixture_likelihood", "mixture_balanced"]

# Baseline 
data = pd.read_csv("../data_4_gluino.csv", header=None).to_numpy()[:n_train,:]
sqrts = data[0,0] + data[0,4] + data[0,8] + data[0,12]
gluino_mass = np.sqrt(data[0,0]**2 - data[0,1]**2 - data[0,2]**2 - data[0,3]**2)
data_continuous = data[:,:16]
data_continuous = torch.tensor(data_continuous.reshape(data_continuous.shape[0], 4, 4))
data_discrete = torch.tensor(data[:,16:], dtype=torch.long)

permute_classes_continuous  = np.array([[0],[1],[2],[3]])   
permute_classes_discrete    = np.array([[2,7],[3,8],[4,9],[5,10]])
permuter = IteratedPermutation(permute_classes_continuous, permute_classes_discrete)

data_continuous_list = []
data_discrete_list = []
while True:
    permuted_data_cont, permute_classes_disc, _ = permuter.forward(data_continuous, data_discrete)
    data_continuous_list.append(permuted_data_cont)
    data_discrete_list.append(permute_classes_disc)

    # Advance permuter
    if not permuter.next():
        break

data_continuous_permuted = torch.cat(data_continuous_list)
data_discrete_permuted = torch.cat(data_discrete_list)

with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)

hyperparams["data_dir"] = os.getcwd() + "/.."

# Copy of the hyperparameters in case of argmax models, as they are different (see discrete_experiments.py)
hyperparams_argmax = dict(hyperparams)
hyperparams_argmax["n_made_units_per_dim"] = 4
hyperparams_argmax["n_RQS_knots"] = 8

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_device', type=int)
parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--n_batches', type=int, default=10)
args = parser.parse_args()

if args.gpu_device is None:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_device)

hel_configs = torch.tensor([[1,1,1,1,1,1], 
                            [1,1,1,1,1,-1],
                            [1,1,1,1,-1,-1],
                            [1,1,1,-1,-1,-1],
                            [1,1,-1,-1,-1,-1]])

col_configs = torch.tensor([[0,1,2,3,4],
                            [1,0,2,3,4],
                            [2,0,1,3,4],
                            [3,0,1,2,4],
                            [4,0,1,2,3]])

conditional_hel_configs = torch.tensor([[1, 1, -1, 1, -1, 1], # Opposite double flip
                                        [1, 1, -1, 1, 1, -1], # Adjacent double flip
                                        [1, 1, -1, 1, 1, 1]]) # Single flip (all work)

conditional_col_configs = torch.tensor([[1, 3, 0, 2, 4], # Opposite double flip
                                        [2, 0, 1, 3, 4], # Adjacent double flip
                                        [1, 0, 2, 3, 4]]) # Single flip (all work)

x_min_hel = 0
x_max_hel = hel_configs.shape[0]
x_min_col = 0
x_max_col = col_configs.shape[0]
x_min_dpsi = 0
x_max_dpsi = np.pi
x_min_energy = 601
x_max_energy = 950
# ------------------------------ Data histograms -----------------------------
data_hel_hist = Histogram(x_min=x_min_hel, x_max=x_max_hel, n_bins=hel_configs.shape[0])
data_col_hist = Histogram(x_min=x_min_col, x_max=x_max_col, n_bins=col_configs.shape[0])
data_conditional_hel_hists = [Histogram(x_min=x_min_dpsi, x_max=x_max_dpsi, n_bins=25) for _ in range(conditional_hel_configs.shape[0])]
data_conditional_col_hists = [Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=25) for _ in range(conditional_col_configs.shape[0])]

# ---------------- Helicity ----------------
for i, hel in enumerate(hel_configs):
    num_of_hel_config = torch.sum(torch.all(data_discrete_permuted[:,:6] == hel, axis=1)).item()
    data_hel_hist.fill(np.ones(num_of_hel_config)*i + 0.5)

data_hel_hist.normalize(norm=data_hel_hist.num_events/n_train/24)
data_hel_hist.write(path="results/hists/data_helicity.dat")

# ---------------- Color ----------------
for i, col in enumerate(col_configs):
    num_of_col_config = torch.sum(torch.all(data_discrete_permuted[:,6:] == col, axis=1))
    data_col_hist.fill(np.ones(num_of_col_config)*i + 0.5)

data_col_hist.normalize(norm=data_col_hist.num_events/n_train/24)
data_col_hist.write("results/hists/data_color.dat")

# ---------------- Conditional helicity distributions ----------------
for i, hel in enumerate(conditional_hel_configs):
    dpsi_data = dpsi(data_continuous[torch.all(data_discrete[:,:6] == hel, axis=1)])
    data_conditional_hel_hists[i].fill(dpsi_data)
    data_conditional_hel_hists[i].normalize(norm=data_conditional_hel_hists[i].num_events/n_train)
    data_conditional_hel_hists[i].write(path="results/hists/data_conditional_helicity_" + str(i) + ".dat")

# ---------------- Conditional color distributions ----------------
for i, col in enumerate(conditional_col_configs):
    energy_data = data_continuous[torch.all(data_discrete[:,6:] == col, axis=1)].numpy()[:,0,0]
    data_conditional_col_hists[i].fill(energy_data)
    data_conditional_col_hists[i].normalize(norm=data_conditional_col_hists[i].num_events/n_train)
    data_conditional_col_hists[i].write(path="results/hists/data_conditional_color_" + str(i) + ".dat")

# ------------------------------ Model histograms -----------------------------
for perm in ["stochastic", "ordered"]:
    hyperparams["permutation"] = perm
    hyperparams_argmax["permutation"] = perm

    for mode in modes:
        if "mixture" in mode:
            hyperparams["discrete_mode"] = mode
            model = MixtureGluinoModel(hyperparams)
        elif "argmax" in mode:
            hyperparams_argmax["discrete_mode"] = mode
            model = GluinoModel(hyperparams_argmax)
        else:
            hyperparams["discrete_mode"] = mode
            model = GluinoModel(hyperparams)

        model.load_state_dict(torch.load("results/model_" + mode + "_" + perm + ".pt", map_location=torch.device(device)))
        model.eval()

        model_hel_hist = Histogram(x_min=x_min_hel, x_max=x_max_hel, n_bins=hel_configs.shape[0])
        model_col_hist = Histogram(x_min=x_min_col, x_max=x_max_col, n_bins=col_configs.shape[0])
        model_conditional_hel_hists = [Histogram(x_min=x_min_dpsi, x_max=x_max_dpsi, n_bins=25) for _ in range(conditional_hel_configs.shape[0])]
        model_conditional_col_hists = [Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=25) for _ in range(conditional_col_configs.shape[0])]

        with torch.no_grad():
            for k in range(args.n_batches):
                print("perm =", perm, "mode =", mode, "batch=", k)

                samples_continuous, samples_discrete = model.sample(args.batch_size)
                samples_continuous = to_phase_space_4_body(samples_continuous, sqrts, gluino_mass)

                # ---------------- Helicity ----------------
                for i, hel in enumerate(hel_configs):
                    num_of_hel_config = torch.sum(torch.all(samples_discrete[:,:6] == hel, axis=1)).item()
                    model_hel_hist.fill(np.ones(num_of_hel_config)*i + 0.5)


                # ---------------- Color ----------------
                for i, col in enumerate(col_configs):
                    num_of_col_config = torch.sum(torch.all(samples_discrete[:,6:] == col, axis=1))
                    model_col_hist.fill(np.ones(num_of_col_config)*i + 0.5)

                # ---------------- Conditional helicity distributions ----------------
                for i, hel in enumerate(conditional_hel_configs):
                    dpsi_model = dpsi(samples_continuous[torch.all(samples_discrete[:,:6] == hel, axis=1)])
                    model_conditional_hel_hists[i].fill(dpsi_model)

                # ---------------- Conditional color distributions ----------------
                for i, col in enumerate(conditional_col_configs):
                    energy_model = samples_continuous[torch.all(samples_discrete[:,6:] == col, axis=1)].numpy()[:,0,0]
                    model_conditional_col_hists[i].fill(energy_model)

        model_hel_hist.normalize(norm=model_hel_hist.num_events/args.batch_size/args.n_batches)
        model_hel_hist.write(path="results/hists/model_" + mode + "_" + perm + "_helicity.dat")

        model_col_hist.normalize(norm=model_col_hist.num_events/args.batch_size/args.n_batches)
        model_col_hist.write(path="results/hists/model_" + mode + "_" + perm + "_color.dat")

        for i, hist in enumerate(model_conditional_hel_hists):
            hist.normalize(norm=hist.num_events/args.batch_size/args.n_batches)
            hist.write(path="results/hists/model_" + mode + "_" + perm + "_conditional_helicity_" + str(i) + ".dat")

        for i, hist in enumerate(model_conditional_col_hists):
            hist.normalize(norm=hist.num_events/args.batch_size/args.n_batches)
            hist.write(path="results/hists/model_" + mode + "_" + perm + "_conditional_color_" + str(i) + ".dat")



# ------------------------------ Classifier histograms -----------------------------
hyperparams["permutation"] = 'ordered'

model = ClassifierGluinoModel(hyperparams)
model.load("../permutation/results/model_ordered_1M.pt", "results/model_classifier.pt", device)
model.eval()

model_hel_hist = Histogram(x_min=x_min_hel, x_max=x_max_hel, n_bins=hel_configs.shape[0])
model_col_hist = Histogram(x_min=x_min_col, x_max=x_max_col, n_bins=col_configs.shape[0])
model_conditional_hel_hists = [Histogram(x_min=x_min_dpsi, x_max=x_max_dpsi, n_bins=25) for _ in range(conditional_hel_configs.shape[0])]
model_conditional_col_hists = [Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=25) for _ in range(conditional_col_configs.shape[0])]

with torch.no_grad():
    for k in range(args.n_batches):
        print("mode = classifier, batch=", k)

        samples_continuous, samples_discrete = model.sample(args.batch_size)
        samples_continuous = to_phase_space_4_body(samples_continuous, sqrts, gluino_mass)

        # ---------------- Helicity ----------------
        for i, hel in enumerate(hel_configs):
            num_of_hel_config = torch.sum(torch.all(samples_discrete[:,:6] == hel, axis=1)).item()
            model_hel_hist.fill(np.ones(num_of_hel_config)*i + 0.5)


        # ---------------- Color ----------------
        for i, col in enumerate(col_configs):
            num_of_col_config = torch.sum(torch.all(samples_discrete[:,6:] == col, axis=1))
            model_col_hist.fill(np.ones(num_of_col_config)*i + 0.5)

        # ---------------- Conditional helicity distributions ----------------
        for i, hel in enumerate(conditional_hel_configs):
            dpsi_model = dpsi(samples_continuous[torch.all(samples_discrete[:,:6] == hel, axis=1)])
            model_conditional_hel_hists[i].fill(dpsi_model)

        # ---------------- Conditional color distributions ----------------
        for i, col in enumerate(conditional_col_configs):
            energy_model = samples_continuous[torch.all(samples_discrete[:,6:] == col, axis=1)].numpy()[:,0,0]
            model_conditional_col_hists[i].fill(energy_model)

model_hel_hist.normalize(norm=model_hel_hist.num_events/args.batch_size/args.n_batches)
model_hel_hist.write(path="results/hists/model_classifier_ordered_helicity.dat")

model_col_hist.normalize(norm=model_col_hist.num_events/args.batch_size/args.n_batches)
model_col_hist.write(path="results/hists/model_classifier_ordered_color.dat")

for i, hist in enumerate(model_conditional_hel_hists):
    hist.normalize(norm=hist.num_events/args.batch_size/args.n_batches)
    hist.write(path="results/hists/model_classifier_ordered_conditional_helicity_" + str(i) + ".dat")

for i, hist in enumerate(model_conditional_col_hists):
    hist.normalize(norm=hist.num_events/args.batch_size/args.n_batches)
    hist.write(path="results/hists/model_classifier_ordered_conditional_color_" + str(i) + ".dat")