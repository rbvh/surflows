#!/usr/bin/env python

import torch
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.lines import Line2D

import pandas as pd 
import numpy as np
import argparse
import yaml

from surflows.gluino.gluino_models import DropoutGluinoModel
from surflows.gluino.gluino_auxiliary import to_phase_space_4_body, to_phase_space_2_body
from surflows.utils import Histogram

xsec_2_gluino = 4.573364
xsec_4_gluino = 0.00018326
norm_2_gluino = xsec_2_gluino/(xsec_2_gluino + xsec_4_gluino)
norm_4_gluino = xsec_4_gluino/(xsec_2_gluino + xsec_4_gluino)

x_min_pT = 0.001
x_max_pT = 1370
x_min_mass = 1220
x_max_mass = 1770
n_bins = 50

n_train = 1000000

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

data_2_gluino = pd.read_csv("../data_2_gluino.csv", header=None).to_numpy()
data_4_gluino = pd.read_csv("../data_4_gluino.csv", header=None).to_numpy()

mass_gluino = np.sqrt(data_4_gluino[0,0]**2 - data_4_gluino[0,1]**2 - data_4_gluino[0,2]**2 - data_4_gluino[0,3]**2)
sqrts = data_4_gluino[0,0] + data_4_gluino[0,4] + data_4_gluino[0,8] + data_4_gluino[0,12]

with open("hyperparams.yaml") as file:
    hyperparams = yaml.safe_load(file)

pT_data_2_gluino = np.sqrt(data_2_gluino[:,1]**2 + data_2_gluino[:,2]**2)
pT_data_4_gluino = np.sqrt(data_4_gluino[:,1]**2 + data_4_gluino[:,2]**2)
mgg_data = np.sqrt((data_4_gluino[:,0] + data_4_gluino[:,4])**2 - (data_4_gluino[:,1] + data_4_gluino[:,5])**2 - (data_4_gluino[:,2] + data_4_gluino[:,6])**2 - (data_4_gluino[:,3] + data_4_gluino[:,7])**2)

data_pT_2_gluino_hist   = Histogram(x_min=x_min_pT, x_max=x_max_pT, n_bins=n_bins)
data_pT_4_gluino_hist   = Histogram(x_min=x_min_pT, x_max=x_max_pT, n_bins=n_bins)
data_mass_hist          = Histogram(x_min=x_min_mass, x_max=x_max_mass, n_bins=n_bins)

data_pT_2_gluino_hist.fill(pT_data_2_gluino)
data_pT_4_gluino_hist.fill(pT_data_4_gluino)
data_mass_hist.fill(mgg_data)

data_pT_2_gluino_hist.normalize(norm=norm_2_gluino)
data_pT_4_gluino_hist.normalize(norm=norm_4_gluino)
data_mass_hist.normalize(norm=norm_4_gluino)

data_pT_2_gluino_hist.write("results/hists/data_pT_2_gluino.dat")
data_pT_4_gluino_hist.write("results/hists/data_pT_4_gluino.dat")
data_mass_hist.write("results/hists/data_mass.dat")

for perm in ["stochastic", "ordered"]:
    for mode in ["likelihood", "balanced", "biased"]:
        file_name = "results/model_" + perm + "_" + mode + ".pt"

        model = DropoutGluinoModel(hyperparams)
        model.load_state_dict(torch.load(file_name, map_location=torch.device(device)))
        model.eval()
        model.to(device)

        model_pT_2_gluino_hist   = Histogram(x_min=x_min_pT, x_max=x_max_pT, n_bins=n_bins)
        model_pT_4_gluino_hist   = Histogram(x_min=x_min_pT, x_max=x_max_pT, n_bins=n_bins)
        model_mass_hist          = Histogram(x_min=x_min_mass, x_max=x_max_mass, n_bins=n_bins)

        with torch.no_grad():
            for k in range(args.n_batches):
                print("perm =", perm, "batch=", k)

                samples_2_gluino = model.sample_conditional(torch.ones(args.batch_size, device=device).long())[0]
                samples_2_gluino = to_phase_space_2_body(samples_2_gluino, sqrts, mass_gluino)
                samples_2_gluino = samples_2_gluino.cpu().numpy()

                model_pT_2_gluino_hist.fill(np.sqrt(samples_2_gluino[:,0,1]**2 + samples_2_gluino[:,0,2]**2))

                samples_4_gluino = model.sample_conditional(torch.zeros(args.batch_size, device=device).long())[0]
                samples_4_gluino = to_phase_space_4_body(samples_4_gluino, sqrts, mass_gluino)
                samples_4_gluino = samples_4_gluino.cpu().numpy()

                model_pT_4_gluino_hist.fill(np.sqrt(samples_4_gluino[:,0,1]**2 + samples_4_gluino[:,0,2]**2))
                model_mass_hist.fill(np.sqrt((samples_4_gluino[:,0,0] + samples_4_gluino[:,1,0])**2 - (samples_4_gluino[:,0,1] + samples_4_gluino[:,1,1])**2 - (samples_4_gluino[:,0,2] + samples_4_gluino[:,1,2])**2 - (samples_4_gluino[:,0,3] + samples_4_gluino[:,1,3])**2))

            model_pT_2_gluino_hist.normalize(norm=norm_2_gluino)
            model_pT_4_gluino_hist.normalize(norm=norm_4_gluino)
            model_mass_hist.normalize(norm=norm_4_gluino)

            model_pT_2_gluino_hist.write("results/hists/" + perm + "_" + mode + "_pT_2_gluino.dat")
            model_pT_4_gluino_hist.write("results/hists/" + perm + "_" + mode + "_pT_4_gluino.dat")
            model_mass_hist.write("results/hists/" + perm + "_" + mode + "_mass.dat")