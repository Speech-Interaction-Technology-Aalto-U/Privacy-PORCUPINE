import torch
import numpy as np
from scipy.stats import binom
eps = 1e-20


def sfvq_inference(input_data, optimized_codebook,mapping_list, mapping_counter):

    entries = optimized_codebook.shape[0]

    distance = (input_data.unsqueeze(-1) - optimized_codebook.t().unsqueeze(0)).square().sum(dim=1)
    integer_index = distance.argmin(dim=1).clamp(min=1, max=entries - 2)

    cm = optimized_codebook[integer_index - 1]
    cc = optimized_codebook[integer_index]
    cp = optimized_codebook[integer_index + 1]

    reminder_index_m = (((cc - cm) * (input_data - cm)).sum(dim=1) / (cc - cm).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
    reminder_index_p = (((cp - cc) * (input_data - cc)).sum(dim=1) / (cp - cc).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

    xhat_m = ((1 - reminder_index_m) * cm) + (reminder_index_m * cc)
    xhat_p = ((1 - reminder_index_p) * cc) + (reminder_index_p * cp)
    distance_m = (input_data - xhat_m).square().sum(dim=1)
    distance_p = (input_data - xhat_p).square().sum(dim=1)

    offset = (distance_p < distance_m).to(torch.int).squeeze() - 1

    indices_first = integer_index + offset
    indices_first_numpy = indices_first.numpy()
    unique_indices, indices_counts = np.unique(indices_first_numpy, return_counts=True)
    mapping_counter[unique_indices] += indices_counts

    c0 = optimized_codebook[integer_index + offset]
    c1 = optimized_codebook[integer_index + offset + 1]

    reminder_index = (((c1 - c0) * (input_data - c0)).sum(dim=1) / (c1 - c0).square().sum(dim=1)).clamp(min=0., max=1.)

    quantized = c0 + (reminder_index.reshape(-1, 1) * (c1 - c0))
    quantized_numpy = quantized.numpy()
    for idx in range(indices_first_numpy.shape[0]):
        target_idx = indices_first_numpy[idx]
        mapping_list[target_idx].append(quantized_numpy[idx:idx+1])

    return quantized.numpy(), mapping_list, mapping_counter


# ordinary vector quantization function
def vector_quantization(input, codebooks):
    input_t = np.expand_dims(input, axis=-1)
    cb_t = np.expand_dims(codebooks.T, axis=0)
    distances = np.sum(np.square(input_t - cb_t), axis=1)
    indices = np.argmin(distances, axis=1)
    return indices


# function to compute KL divergence
def kl_from_binomial(histogram, num_samples):
    bins = histogram.shape[0]
    bincount = np.bincount(histogram)  # distribution of number of samples
    b = binom(num_samples, 1.0 / bins)
    ix = np.arange(0, len(bincount))
    observation = bincount/np.sum(bincount)
    true_model = b.pmf(ix)

    zeros_indices = np.where(observation == 0.0)[0]
    temp = np.log2(observation / true_model)
    temp[zeros_indices] = 0.0
    kl_div = np.sum(observation * temp)

    return kl_div


# function to compute sparseness (norm2/norm1)
def sparseness(histogram):
    l1 = np.sum(np.abs(histogram))
    l2 = np.sqrt(np.sum(np.square(histogram)))

    return l2/l1


# function to compute average disclosure
def average_disclosure(histogram, num_samples):
    probs = histogram / num_samples
    zero_indices = np.where(probs == 0.0)[0]
    probs2 = probs.copy()
    probs2[zero_indices] = 1e-12

    return -1 * np.sum(probs * np.log2(probs2))

