import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

def plot_hist(data, data_baseline, ax, ax_ratio, label, color='black', linewidth=1):
    counts = data[:,2]
    errors = data[:,3]

    bins = data[:,0]
    bins = np.append(bins, data[-1,1])

    centers = (bins[1:] + bins[:-1])/2

    fill_x = np.array([e for edge in bins for e in [edge, edge]][1:-1])
    fill_delta_y = np.array([y for error in errors for y in [error, error]])
    fill_central_y = np.array([c for count in counts for c in [count, count]])

    # Regular plot
    ax.hist(centers, bins=bins, histtype='step', label=label, color=color, linewidth=linewidth, weights=counts)
    ax.fill_between(fill_x, fill_central_y - fill_delta_y, fill_central_y + fill_delta_y, facecolor=color, alpha=0.3)
    
    # Ratio plot
    counts_baseline = data_baseline[:,2]
    ax_ratio.hist(centers, bins=bins, histtype='step', color=color, linewidth=linewidth, weights=counts/counts_baseline)

    # Errors in ratio
    ratio_counts = counts/counts_baseline
    ratio_counts[np.isinf(ratio_counts)] = 1
    ratio_errors = errors/counts_baseline
    ratio_errors[np.isinf(ratio_errors)] = 1
    
    fill_central_y = np.array([c for count in ratio_counts for c in [count, count]])
    fill_delta_y = np.array([y for error in ratio_errors for y in [error, error]])
    ax_ratio.fill_between(fill_x, fill_central_y - fill_delta_y, fill_central_y + fill_delta_y, facecolor=color, alpha=0.3)

    # Adjust x lims
    # ax.set_xlim(xmin=bins[0]+1, xmax=bins[-1]-1)

textwidth = 9
textheight = textwidth/2.4
fontsize = 10.95

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
# modes = ["uniform_dequantization", "flow_dequantization", "uniform_argmax", "flow_argmax", "mixture_likelihood", "mixture_balanced"]
# labels = ["Uniform dequantization", "Flow dequantization", "Uniform argmax", "Flow argmax", "Mixture likelihood", "Mixture balanced"]

modes = ["flow_dequantization", "flow_argmax", "mixture_likelihood", "classifier"]
labels = ["Dequantization (flow)", "Argmax (flow)", "Mixture (likelihood)", "Classifier"]

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 4.0
mpl.rcParams['xtick.minor.size'] = 2.0
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.5

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.major.size'] = 4.0
mpl.rcParams['ytick.minor.size'] = 2.0
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['ytick.minor.width'] = 0.5

mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams["legend.labelspacing"] = 0.1
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["legend.handletextpad"] = 0.5
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
#mpl.rcParams["legend.columnspacing"] = 0
mpl.rcParams.update({'font.size': fontsize})

# ------------------------------ Discrete distributions -----------------------------
data_helicity_hist = np.genfromtxt("results/hists/data_helicity.dat")
data_color_hist = np.genfromtxt("results/hists/data_color.dat")

for perm in ["ordered", "stochastic"]:
    fig, axs = plt.subplots(2, 2, sharex='col', gridspec_kw={'hspace': 0, 'wspace': 0.25, 'height_ratios': [5,2]})
    fig.set_size_inches(textwidth, textheight)

    plot_hist(data_helicity_hist, data_helicity_hist, axs[0,0], axs[1,0], label='Train', linewidth=1.5)
    plot_hist(data_color_hist, data_color_hist, axs[0,1], axs[1,1], label='Train', linewidth=1.5)

    for i, mode in enumerate(modes):
        if mode == "classifier" and perm == "stochastic":
            continue
        model_helicity_hist = np.genfromtxt("results/hists/model_" + mode + "_" + perm + "_helicity.dat")
        model_color_hist = np.genfromtxt("results/hists/model_" + mode + "_" + perm + "_color.dat")

        plot_hist(model_helicity_hist, data_helicity_hist, axs[0,0], axs[1,0], label=labels[i], color=colors[i])
        plot_hist(model_color_hist, data_color_hist, axs[0,1], axs[1,1], label=labels[i], color=colors[i])

    axs[0,0].set_xlim(0.01, 5 - 0.01)
    axs[0,0].set_ylim(0.001, 0.085)
    axs[1,0].set_ylim(0.8, 1.2)

    axs[1,0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    axs[1,0].set_xticklabels([r'$(+,+) \to (+,+,+,+)$', r'$(+,+) \to (+,+,+,-)$', r'$(+,+) \to (+,+,-,-)$', r'$(+,+) \to (+,-,-,-)$', r'$(+,+) \to (-,-,-,-)$'], rotation=90, fontsize=6.2)
    axs[1,0].minorticks_off()

    axs[0,0].set_title(r'Helicity')
    axs[0,0].set_ylabel(r'$\sigma_{\mathrm{hel}} / \sigma_{\mathrm{tot}}$')
    axs[1,0].set_ylabel(r'Ratio')

    axs[0,1].set_xlim(0.01, 5 - 0.01)
    axs[0,1].set_ylim(0.001, 0.021)
    axs[1,1].set_ylim(0.8, 1.2)

    axs[1,1].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    axs[1,1].set_xticklabels([r'$(1,2) \to (3,4,5,6)$', r'$(1,3) \to (2,4,5,6)$', r'$(1,4) \to (2,3,5,6)$', r'$(1,5) \to (2,3,4,6)$', r'$(1,6) \to (2,3,4,5)$'], rotation=90, fontsize=9)
    axs[1,1].minorticks_off()

    axs[0,1].set_title(r'Colour')
    axs[0,1].set_ylabel(r'$\sigma_{\mathrm{col}} / \sigma_{\mathrm{tot}}$')
    axs[1,1].set_ylabel(r'Ratio')

    plt.axes(axs[0,0])
    leg_handles, leg_labels = axs[0,0].get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in leg_handles]
    plt.legend(handles=new_handles, labels=leg_labels, loc='upper right')

    plt.savefig("results/plots/discrete_" + perm + ".pdf", bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# ------------------------------ Conditional helicity distributions -----------------------------
for perm in ["ordered", "stochastic"]:
    fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [5,2]})
    fig.set_size_inches(textwidth, textheight)

    for i in range(3):
        data_conditional_helicity_hist = np.genfromtxt("results/hists/data_conditional_helicity_" + str(i) + ".dat")
        plot_hist(data_conditional_helicity_hist, data_conditional_helicity_hist, axs[0,i], axs[1,i], label=r'Train', linewidth=1.5)

        for j, mode in enumerate(modes):
            if mode == "classifier" and perm == "stochastic":
                continue
            model_conditional_helicity_hist = np.genfromtxt("results/hists/model_" + mode + "_" + perm + "_conditional_helicity_" + str(i) + ".dat")
            plot_hist(model_conditional_helicity_hist, data_conditional_helicity_hist, axs[0,i], axs[1,i], label=labels[j], color=colors[j])

        axs[0,i].set_ylim(0.00001, 0.0149)
        axs[1,i].set_xlim(0.01, np.pi - 0.01)
        axs[1,i].set_ylim(0.5, 1.5)

        axs[0,i].set_xticks([np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        axs[0,i].set_xticklabels([r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])

        axs[1,i].set_xlabel(r'$\Delta \psi_{(12)3}$')

    axs[0,0].set_title(r'$(+,+) \to (-,+,-,+)$')
    axs[0,1].set_title(r'$(+,+) \to (-,+,+,-)$')
    axs[0,2].set_title(r'$(+,+) \to (-,+,+,+)$')

    axs[0,0].set_ylabel(r'$\frac{1}{\sigma_{\mathrm{tot}}} \frac{d\sigma}{d\Delta \psi_{12}}$')
    axs[1,0].set_ylabel(r'Ratio')

    plt.axes(axs[0,0])
    leg_handles, leg_labels = axs[0,0].get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in leg_handles]
    plt.legend(handles=new_handles, labels=leg_labels, loc='upper right')
    
    plt.savefig("results/plots/conditional_helicity_" + perm + ".pdf", bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
        
# ------------------------------ Conditional color distributions -----------------------------
for perm in [ "ordered", "stochastic"]:
    fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [5,2]})
    fig.set_size_inches(textwidth, textheight)   

    for i in range(3):
        data_conditional_color_hist = np.genfromtxt("results/hists/data_conditional_color_" + str(i) + ".dat")
        plot_hist(data_conditional_color_hist, data_conditional_color_hist, axs[0,i], axs[1,i], label=r'Train', linewidth=1.5)

        for j, mode in enumerate(modes):
            if mode == "classifier" and perm == "stochastic":
                continue
            model_conditional_color_hist = np.genfromtxt("results/hists/model_" + mode + "_" + perm + "_conditional_color_" + str(i) + ".dat")
            plot_hist(model_conditional_color_hist, data_conditional_color_hist, axs[0,i], axs[1,i], label=labels[j], color=colors[j])

        axs[0,i].set_ylim(0.00001, 0.00009)
        axs[1,i].set_xlim(602, 949)
        axs[1,i].set_ylim(0.5, 1.5)
        axs[1,i].set_xlabel(r'$E_{\tilde{g}_1}$ [GeV]')

    axs[0,0].set_title(r'$(1,3) \to (5,2,4,6)$')
    axs[0,1].set_title(r'$(1,4) \to (2,3,5,6)$')
    axs[0,2].set_title(r'$(1,3) \to (2,4,5,6)$')

    axs[0,0].set_ylabel(r'$\frac{1}{\sigma_{\mathrm{tot}}} \frac{d\sigma}{dE_{\tilde{g}_1}} \mathrm{ [GeV]}^{-1}$')
    axs[1,0].set_ylabel(r'Ratio')

    plt.axes(axs[0,0])
    leg_handles, leg_labels = axs[0,0].get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in leg_handles]
    plt.legend(handles=new_handles, labels=leg_labels, loc='upper right')

    plt.savefig("results/plots/conditional_color_" + perm + ".pdf", bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()