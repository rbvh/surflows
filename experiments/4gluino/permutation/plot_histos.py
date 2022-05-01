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
    ax.set_xlim(xmin=bins[0]+1, xmax=bins[-1]-1)


textwidth = 9
textheight = textwidth/2.4
fontsize = 10.95

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

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


# ------------------------------------------------ Energy plots ------------------------------------------------

labels = ["Flow (gluino 1)", "Flow (gluino 2)", "Flow (gluino 3)", "Flow (gluino 4)"]

for data in ["50k", "100k", "200k", "1M"]:
    fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [5,2]})
    fig.set_size_inches(textwidth, textheight)

    data_hist = np.genfromtxt("results/hists/data_" + data + "_energy.dat")
    plot_hist(data_hist, data_hist, axs[0,0], axs[1,0], label='Train', linewidth=1.5)
    plot_hist(data_hist, data_hist, axs[0,1], axs[1,1], label='Train', linewidth=1.5)
    plot_hist(data_hist, data_hist, axs[0,2], axs[1,2], label='Train', linewidth=1.5)

    for i, perm in enumerate(["without_perm", "stochastic", "ordered"]):
        for j in range(4):
            model_hist = np.genfromtxt('results/hists/' + perm + "_" + data + "_energy_" + str(j) + ".dat")
            plot_hist(model_hist, data_hist, axs[0,i], axs[1,i], label=labels[j], color=colors[j])

        axs[0,i].set_ylim(ymin=1e-4, ymax=0.0049)        
        axs[1,i].set_ylim(ymin=0.8, ymax=1.2)
        axs[1,i].set_xlabel(r'$E_{\tilde{g}} $ [GeV]')  
    
    axs[0,0].set_ylabel(r'$\frac{1}{\sigma} \frac{d\sigma}{dE_{\tilde{g}}} \mathrm{[GeV]}^{-1}$')
    axs[1,0].set_ylabel(r'Flow/MC')

    axs[0,0].set_title("No permutation")
    axs[0,1].set_title("Stochastic permutation")
    axs[0,2].set_title("Sort surjection")

    plt.axes(axs[0,0])
    old_handles, old_labels = axs[0,0].get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in old_handles]
    plt.legend(handles=new_handles, labels=old_labels, loc='upper left', bbox_to_anchor=(0.01,0.5))

    plt.savefig("results/plots/energies_" + data + ".pdf", bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# ------------------------------------------------- Mass plots -------------------------------------------------
fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [5,2]})
fig.set_size_inches(textwidth, textheight)

data_hist = np.genfromtxt('results/hists/data_1M_mass.dat')

for i, perm in enumerate(["without_perm", "stochastic", "ordered"]):
    plot_hist(data_hist, data_hist, axs[0,i], axs[1,i], label='Train', linewidth=1.5)

    for j, data in enumerate(["50k", "200k", "1M"]):
        model_hist = np.genfromtxt('results/hists/' + perm + "_" + data + "_mass.dat")
        plot_hist(model_hist, data_hist, axs[0,i], axs[1,i], label=data, color=colors[j])

    axs[0,i].set_ylim(ymin=1e-6, ymax=0.0046)
    axs[1,i].set_ylim(ymin=0.8, ymax=1.2)
    axs[1,i].set_xlabel(r'$m_{\tilde{g}\tilde{g}} $ [GeV]')
    
axs[0,0].set_title("No permutation")
axs[0,1].set_title("Stochastic permutation")
axs[0,2].set_title("Sort surjection")

axs[0,0].set_ylabel(r'$\frac{1}{\sigma} \frac{d\sigma}{dm_{\tilde{g} \tilde{g}}} \mathrm{[GeV]}^{-1}$')
axs[1,0].set_ylabel(r'Flow/MC')

plt.axes(axs[0,0])
old_handles, old_labels = axs[0,0].get_legend_handles_labels()
new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in old_handles]
plt.legend(handles=new_handles, labels=old_labels, loc='upper left', bbox_to_anchor=(0.01,0.4))

plt.savefig("results/plots/mass.pdf", bbox_inches='tight')
plt.close()