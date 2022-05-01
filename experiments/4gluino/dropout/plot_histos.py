import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

def plot_hist(data, data_baseline, ax, ax_ratio, label=None, color='black', linewidth=1, linestyle='solid'):
    counts = data[:,2]
    errors = data[:,3]

    bins = data[:,0]
    bins = np.append(bins, data[-1,1])

    centers = (bins[1:] + bins[:-1])/2

    fill_x = np.array([e for edge in bins for e in [edge, edge]][1:-1])
    fill_delta_y = np.array([y for error in errors for y in [error, error]])
    fill_central_y = np.array([c for count in counts for c in [count, count]])

    # Regular plot
    ax.hist(centers, bins=bins, histtype='step', label=label, color=color, linewidth=linewidth, linestyle=linestyle, weights=counts)
    ax.fill_between(fill_x, fill_central_y - fill_delta_y, fill_central_y + fill_delta_y, facecolor=color, alpha=0.3)
    
    # Ratio plot
    counts_baseline = data_baseline[:,2]
    ax_ratio.hist(centers, bins=bins, histtype='step', color=color, linewidth=linewidth, linestyle=linestyle, weights=counts/counts_baseline)

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
labels = ["Likelihood", "Balanced", "Biased"]

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

data_pT_2_gluino = np.genfromtxt("results/hists/data_pT_2_gluino.dat")
data_pT_4_gluino = np.genfromtxt("results/hists/data_pT_4_gluino.dat")
data_mass = np.genfromtxt("results/hists/data_mass.dat")

for perm in ["stochastic", "ordered"]:
    fig, axs = plt.subplots(3, 2, sharex='col', gridspec_kw={'hspace': 0, 'height_ratios': [5,1,1]})
    gs = axs[1, 1].get_gridspec()
    axs[1,1].remove()
    axs[2,1].remove()
    axtwo = fig.add_subplot(gs[1:, -1], sharex=axs[0,1])

    fig.add_subplot
    fig.set_size_inches(textwidth, textheight)

    plot_hist(data_pT_4_gluino, data_pT_4_gluino, axs[0,0], axs[2,0], label='Train', color="black", linewidth=1.5, linestyle='solid')
    plot_hist(data_pT_2_gluino, data_pT_2_gluino, axs[0,0], axs[1,0], color="black", linewidth=1.5, linestyle='dashed')
    plot_hist(data_mass, data_mass, axs[0,1], axtwo, color='black', linewidth=1.5, linestyle='solid')

    for i, mode in enumerate(["likelihood", "balanced"]):

        model_pT_2_gluino = np.genfromtxt("results/hists/" + perm + "_" + mode + "_pT_2_gluino.dat")
        model_pT_4_gluino = np.genfromtxt("results/hists/" + perm + "_" + mode + "_pT_4_gluino.dat")
        model_mass = np.genfromtxt("results/hists/" + perm + "_" + mode + "_mass.dat")

        plot_hist(model_pT_4_gluino, data_pT_4_gluino, axs[0,0], axs[2,0], label=labels[i], color=colors[i], linewidth=1., linestyle='solid')
        plot_hist(model_pT_2_gluino, data_pT_2_gluino, axs[0,0], axs[1,0], color=colors[i], linewidth=1., linestyle='dashed')
        plot_hist(model_mass, data_mass, axs[0,1], axtwo, color=colors[i], linewidth=1., linestyle='solid')

    axs[0,0].set_yscale('log')
    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    axs[0,0].yaxis.set_minor_locator(y_minor)
    axs[0,0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    axs[1,0].set_ylim(0.8, 1.2)
    axs[2,0].set_ylim(0.8, 1.19999)

    axs[2,0].set_xlabel(r'$p_{\perp} $ [GeV]')
    axs[0,0].set_ylabel(r'$\frac{1}{\sigma_{\mathrm{sum}}} \frac{d\sigma}{d p_{\perp}} \mathrm{[GeV]}^{-1}$')

    axs[0,1].set_ylim(ymin=1e-10, ymax=1.9e-7)
    axtwo.set_ylim(0.8, 1.2)

    axtwo.set_xlabel(r'$m_{\tilde{g}\tilde{g}}$ [GeV]')
    axs[0,1].set_ylabel(r'$\frac{1}{\sigma_{\mathrm{sum}}} \frac{d\sigma}{m_{\tilde{g}\tilde{g}}} \mathrm{[GeV]}^{-1}$')

    handles_hist_1, labels_hist_1 = axs[0,0].get_legend_handles_labels()
    new_handles_hist_1 = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles_hist_1]
    axs[0,1].legend(handles=new_handles_hist_1, labels=labels_hist_1, loc='upper right')

    labels_hist_2 = ['4 gluino', '2 gluino']
    handles_hist_2 = [Line2D([],[],c='black',ls='solid'), Line2D([],[],c='black',ls='dashed')]
    axs[0,0].legend(handles_hist_2, labels_hist_2, loc='lower right')


    plt.savefig("results/plots/" + perm + ".pdf", bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()