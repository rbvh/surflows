#!/usr/bin/env python

import torch
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.lines import Line2D
import argparse
import yaml
from ppflows.gluino.gluino_models import GluinoModel
from ppflows.utils import Histogram

import pandas as pd 
import numpy as np

from ppflows.gluino.gluino_auxiliary import to_phase_space_4_body, to_angle_4_body

n_train = 1000000

x_min_mass = 1220
x_max_mass = 1770
x_min_energy = 601
x_max_energy = 999
n_bins = 50

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

# Get the baseline
data = pd.read_csv("../data_4_gluino.csv", header=None).to_numpy()
sqrts = data[0,0] + data[0,4] + data[0,8] + data[0,12]
gluino_mass = np.sqrt(data[0,0]**2 - data[0,1]**2 - data[0,2]**2 - data[0,3]**2)
data_mass = np.sqrt((data[:,0] + data[:,4])**2 - (data[:,1] + data[:,5])**2 - (data[:,2] + data[:,6])**2 - (data[:,3] + data[:,7])**2)

with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)

for i, train_num in enumerate(["50k", "100k", "200k", "1M"]):
    if train_num == "50k":
        n_train_num = 50000
    if train_num == "100k":
        n_train_num = 100000
    if train_num == "200k":
        n_train_num = 200000
    if train_num == "500k":
        n_train_num = 500000
    if train_num == "1M":
        n_train_num = 1000000

    data_energy_hist = Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=n_bins)
    data_energy_hist.fill(data[:n_train_num, 0])
    data_energy_hist.write("results/hists/data_" + train_num + "_energy.dat")

    data_angle_hist = Histogram(x_min=-1, x_max=1, n_bins=n_bins)
    data_angles = to_angle_4_body(torch.tensor(data[:,:16].reshape(-1,4,4)))
    data_angle_hist.fill(data_angles[:,0]*2 - 1)
    data_angle_hist.write("results/hists/data_" + train_num + "_angle.dat")

    data_mass_hist = Histogram(x_min=x_min_mass, x_max=x_max_mass, n_bins=n_bins)
    data_mass_hist.fill(data_mass[:n_train_num])
    data_mass_hist.write("results/hists/data_" + train_num + "_mass.dat")
    
for i, perm in enumerate(["none", "stochastic", "ordered"]):
    hyperparams["permutation"] = perm

    for j, train_num in enumerate(["50k", "100k", "200k", "1M"]):
        model_file_name = "results/model_" + perm + "_" + train_num + ".pt"
        model = GluinoModel(hyperparams)
        model.load_state_dict(torch.load(model_file_name, map_location=torch.device(device)))
        model.eval()
        model.to(device)

        model_mass_hist     = Histogram(x_min=x_min_mass, x_max=x_max_mass, n_bins=n_bins)
        model_energy_0_hist = Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=n_bins)
        model_energy_1_hist = Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=n_bins)
        model_energy_2_hist = Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=n_bins)
        model_energy_3_hist = Histogram(x_min=x_min_energy, x_max=x_max_energy, n_bins=n_bins)
        model_angle_0_hist  = Histogram(x_min=-1, x_max=1, n_bins=n_bins)
        model_angle_1_hist  = Histogram(x_min=-1, x_max=1, n_bins=n_bins)
        model_angle_2_hist  = Histogram(x_min=-1, x_max=1, n_bins=n_bins)
        model_angle_3_hist  = Histogram(x_min=-1, x_max=1, n_bins=n_bins)

        with torch.no_grad():
            for k in range(args.n_batches):
                print("perm =", perm, "train_num=", train_num, "batch=", k)

                samples, _ = model.sample(args.batch_size)
                samples_phase_space = to_phase_space_4_body(samples, sqrts, gluino_mass)
                samples_mass = np.sqrt((samples_phase_space[:,0,0] + samples_phase_space[:,1,0])**2 - (samples_phase_space[:,0,1] + samples_phase_space[:,1,1])**2 - (samples_phase_space[:,0,2] + samples_phase_space[:,1,2])**2 - (samples_phase_space[:,0,3] + samples_phase_space[:,1,3])**2)

                model_mass_hist.fill(samples_mass)
                model_energy_0_hist.fill(samples_phase_space[:,0,0])
                model_energy_1_hist.fill(samples_phase_space[:,1,0])
                model_energy_2_hist.fill(samples_phase_space[:,2,0])
                model_energy_3_hist.fill(samples_phase_space[:,3,0])
                model_angle_0_hist.fill(samples[:,0]*2 - 1)
                model_angle_1_hist.fill(samples[:,2]*2 - 1)
                model_angle_2_hist.fill(samples[:,4]*2 - 1)
                model_angle_3_hist.fill(samples[:,6]*2 - 1)

        model_mass_hist.write("results/hists/" + perm + "_" + train_num + "_mass.dat")
        model_energy_0_hist.write("results/hists/" + perm + "_" + train_num + "_energy_0.dat")
        model_energy_1_hist.write("results/hists/" + perm + "_" + train_num + "_energy_1.dat")
        model_energy_2_hist.write("results/hists/" + perm + "_" + train_num + "_energy_2.dat")
        model_energy_3_hist.write("results/hists/" + perm + "_" + train_num + "_energy_3.dat")
        model_angle_0_hist.write("results/hists/" + perm + "_" + train_num + "_angle_0.dat")
        model_angle_1_hist.write("results/hists/" + perm + "_" + train_num + "_angle_1.dat")
        model_angle_2_hist.write("results/hists/" + perm + "_" + train_num + "_angle_2.dat")
        model_angle_3_hist.write("results/hists/" + perm + "_" + train_num + "_angle_3.dat")