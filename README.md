[![arXiv](https://img.shields.io/badge/arXiv-2205.01697-<COLOR>.svg)](https://arxiv.org/abs/2205.01697)

`surflows` is the public code associated with [Event Generation and Density Estimation with Surjective Normalizing Flows](https://arxiv.org/abs/2205.01697)

## Installation
```
cd surflows 
pip3 install -e surflows
```

## Usage
# Four gluino experiments
Scripts can be found in `experiments/4gluino`. The script `download_data.sh` can be used to retrieve the training data. Experiments for permutation invariance, varying number of objects and discrete features can be found in `permutation`, `dropout` and `discrete` respectively. All directories have scripts `xxx_experiments.py` that train the flow, with the option of running on a GPU, as well as `make_histos.py` and `plot_histos.py` to produce the plots in the paper.

# Dark Machines anomaly detection
Scripts can be found in `experiments/darkmachines`. The script `download_data.sh` can be used to retrieve the training data. The script `anomaly_detection_experiments.py` can be used to train a variety of models with different types of handling of permutation symmetry and discrete features.

## References
The code in `surflows/rqs_flow` is a modified and trimmed-down version of [bayesiains/nflows](https://github.com/bayesiains/nflows).