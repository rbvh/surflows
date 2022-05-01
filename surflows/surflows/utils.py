import numpy as np
import torch


def tile(x, n):
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nets.Module
    :param model: nets.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.

    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features):
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.

    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )
    mask[indices] += 1
    return mask


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value, bound=1 - 1e-3):
    """
    For a dataset with max value 'max_value', returns the temperature such that

        sigmoid(temperature * max_value) = bound.

    If temperature is greater than 1, returns 1.

    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(-(1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature


def gaussian_kde_log_eval(samples, query):
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std ** 2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)

def exclusive_rand(*size, device=None):
    """
    Generate uniform (0,1), instead of the [0,1) of torch
    """
    rand = torch.rand(size, device=device)
    while (rand == 0.).any():
        rand[rand == 0.] = torch.rand(rand[rand == 0.].shape, device=device)
    return rand

class EarlyStopper:
    """
    Early stopping for SGD 
    """
    def __init__(self, patience=20, delta=0, mode = 'min'):
        
        self.patience_ = patience
        self.delta_ = delta
        self.mode_ = mode

        self.best_loss_ = None
        self.counter_ = 0
        self.early_stop_ = False
    
    def __call__(self, val_loss):
        if self.best_loss_ == None:
            # The first loss we get
            self.best_loss_ = val_loss
        elif self.is_better(val_loss):
            # This loss is the best so far - Reset counter and replace
            self.best_loss_ = val_loss
            self.counter_ = 0
        else:
            # This loss is worse than the best
            self.counter_ += 1 
            if self.counter_ >= self.patience_:
                self.early_stop_ = True

        return self.early_stop_

    def is_better(self, val_loss):
        if self.mode_ == "min":
            return val_loss < self.best_loss_ - self.delta_
        elif self.mode_ == "max":
            return val_loss > self.best_loss_ + self.delta_
        else:
            raise ValueError("{} is not a valid mode".format(self.mode_))

class Histogram:
    def __init__(self, x_min, x_max, n_bins):
        self.x_min = x_min
        self.x_max = x_max
        self.n_bins = n_bins

        self.bins = np.linspace(x_min, x_max, n_bins+1)
        self.bin_width = (x_max - x_min)/n_bins

        self.counts = np.zeros(n_bins)
        self.errors = np.zeros(n_bins)

        self.num_events = 0

        self.is_normalized = False

    def fill(self, x):
        new_counts, _ = np.histogram(x, bins=self.bins)
        self.num_events += x.shape[0]

        self.counts += new_counts
        self.errors = self.counts ** 0.5

    def normalize(self, norm=1):
        assert self.is_normalized is False, "Tried to normalize a normalized histogram"

        self.is_normalized = True

        self.counts *= norm/self.num_events/self.bin_width
        self.errors *= norm/self.num_events/self.bin_width

    def write(self, path):
        # Normalize the histogram if it has not been already
        if self.is_normalized is False:
            self.normalize()

        x_left = self.bins[:-1]
        x_right = self.bins[1:]

        np.savetxt(path, np.stack((x_left, x_right, self.counts, self.errors), axis=1))