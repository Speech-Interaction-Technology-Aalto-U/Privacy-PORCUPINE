import numpy as np
from evaluation_utils import sfvq_inference, vector_quantization, kl_from_binomial, sparseness, average_disclosure
import torch
import math
import itertools
import os

print('*************************************************************')
print("Ignore 'RuntimeWarning: divide by zero encountered in log2' \nThese values will be multiplied by zero in KL div computation!")
print('*************************************************************')

bitrate_list = [4,5,6] # bitrates you trained VQ and SFVQ
run_index_list = [1,2,3,4,5] # run index (to plot the confidence intervals for the results you need to train each
                             # training configuration for several times

num_bitrates = len(bitrate_list)
num_runs = len(run_index_list) # number of experiments for each individual bitrate

# define arrays to store the computed metrics
kl_div_array_vq = np.zeros((num_bitrates, num_runs))
kl_div_array_sfvq = np.zeros((num_bitrates, num_runs))

std_array_vq = np.zeros((num_bitrates, num_runs))
std_array_sfvq = np.zeros((num_bitrates, num_runs))

sparseness_array_vq = np.zeros((num_bitrates, num_runs))
sparseness_array_sfvq = np.zeros((num_bitrates, num_runs))

worst_case_disclosure_array_vq = np.zeros((num_bitrates, num_runs))
worst_case_disclosure_array_sfvq = np.zeros((num_bitrates, num_runs))

average_disclosure_vq = np.zeros((num_bitrates, num_runs))
average_disclosure_sfvq = np.zeros((num_bitrates, num_runs))

# address to load trained codebooks
address_vq = './codebooks_vq/'
address_sfvq = './codebooks_sfvq/'
# address to save final computed evaluation metrics
save_address = "evaluation_metrics"
os.makedirs(save_address, exist_ok=True)

# train data samples are used for computing the resampled codebooks for SFVQ and then we evaluate with test data
data_train = np.load('train_vectors.npy')
num_data_samples = data_train.shape[0]
data_dim = data_train.shape[1]
test_data = np.load('test_vectors.npy')
num_test_data_samples = test_data.shape[0]

loading_batch_size = 5000 # number of train set data samples to be processed in a batch (does not effect final results)
if num_data_samples % loading_batch_size==0:
    num_batches = int(np.floor(num_data_samples/loading_batch_size))
else:
    num_batches = int(np.floor(num_data_samples / loading_batch_size) + 1)



for bitrate_idx, bitrate in enumerate(bitrate_list):

    for run_idx in run_index_list:

        print(f'bitrate: {bitrate} | run_idx: {run_idx}')

        codebooks_vq = np.load(address_vq + f'codebook_{bitrate}bit_r{run_idx}.npy')
        codebooks_sfvq = np.load(address_sfvq + f'codebook_{bitrate}bit_r{run_idx}.npy')
        entries = codebooks_sfvq.shape[0]

        mapping_list = [[] for _ in range(entries-1)]
        mapping_counter = np.zeros((entries-1,), np.int32)

        for batch_idx in range(num_batches):
            if batch_idx == (num_batches - 1):
                data_batch = data_train[batch_idx * loading_batch_size:]
            else:
                data_batch = data_train[batch_idx * loading_batch_size: (batch_idx + 1) * loading_batch_size]

            mapped_train_data, mapping_list, mapping_counter = sfvq_inference(torch.from_numpy(data_batch), torch.from_numpy(codebooks_sfvq), mapping_list, mapping_counter)

        mapping_list_final = list(itertools.chain.from_iterable(mapping_list))
        stacked_mappings = np.vstack(mapping_list_final)

        num_picking_points = num_data_samples / entries
        temp = math.modf(num_picking_points)
        pick_points_int = int(temp[1])
        pick_points_float = temp[0]

        counter = 0
        resampled_codebooks = np.zeros_like(codebooks_sfvq)
        for i in range(entries):
            temp = stacked_mappings[counter:counter+pick_points_int+1]
            counter += pick_points_int

            chunk1 = temp[0:pick_points_int,:]
            chunk2 = temp[-1:, :] * pick_points_float

            total_chunk = np.concatenate((chunk1, chunk2), axis=0)

            mean_vectors = np.mean(total_chunk, axis=0, keepdims=True)
            resampled_codebooks[i] = mean_vectors


        vq_cb_indices = vector_quantization(test_data, codebooks_vq)
        sfvq_cb_indices = vector_quantization(test_data, resampled_codebooks)

        vq_cb_hist = np.histogram(vq_cb_indices, bins=entries)[0]
        sfvq_cb_hist = np.histogram(sfvq_cb_indices, bins=entries)[0]

        kl_div_array_vq[bitrate_idx, run_idx - 1] = kl_from_binomial(vq_cb_hist, num_test_data_samples)
        kl_div_array_sfvq[bitrate_idx, run_idx - 1] = kl_from_binomial(sfvq_cb_hist, num_test_data_samples)

        std_array_vq[bitrate_idx, run_idx - 1] = np.std(vq_cb_hist)
        std_array_sfvq[bitrate_idx, run_idx - 1] = np.std(sfvq_cb_hist)

        sparseness_array_vq[bitrate_idx, run_idx - 1] = sparseness(vq_cb_hist)
        sparseness_array_sfvq[bitrate_idx, run_idx - 1] = sparseness(sfvq_cb_hist)

        worst_case_disclosure_array_vq[bitrate_idx, run_idx-1] = -1 * np.log2((np.min(vq_cb_hist)+1) / np.sum(vq_cb_hist))
        worst_case_disclosure_array_sfvq[bitrate_idx, run_idx-1] = -1 * np.log2((np.min(sfvq_cb_hist)+1) / np.sum(sfvq_cb_hist))

        average_disclosure_vq[bitrate_idx, run_idx - 1] = average_disclosure(vq_cb_hist, num_test_data_samples)
        average_disclosure_sfvq[bitrate_idx, run_idx - 1] = average_disclosure(sfvq_cb_hist, num_test_data_samples)

    print('##############################')


np.save(save_address + f'average_disc_vq', average_disclosure_vq)
np.save(save_address + f'worst_case_disc_vq', worst_case_disclosure_array_vq)
np.save(save_address + f'std_vq', std_array_vq)
np.save(save_address + f'kl_div_vq', kl_div_array_vq)
np.save(save_address + f'sparseness_vq', sparseness_array_vq)

np.save(save_address + f'average_disc_sfvq', average_disclosure_sfvq)
np.save(save_address + f'worst_case_disc_sfvq', worst_case_disclosure_array_sfvq)
np.save(save_address + f'std_sfvq', std_array_sfvq)
np.save(save_address + f'kl_div_sfvq', kl_div_array_sfvq)
np.save(save_address + f'sparseness_sfvq', sparseness_array_sfvq)

#################################################################################
import matplotlib.pyplot as plt
import matplotlib
text_size = 18
legend_size = 4
font = {'family' : 'Times New Roman', 'size' : text_size}
matplotlib.rc('font', **font)
figsize = (16,10)

kl_div_vq_mean = np.mean(kl_div_array_vq, axis=1)
std_vq_mean = np.mean(std_array_vq, axis=1)
worst_case_disc_vq_mean = np.mean(worst_case_disclosure_array_vq, axis=1)
sparseness_vq_mean = np.mean(sparseness_array_vq, axis=1)
avg_disc_vq_mean = np.mean(average_disclosure_vq, axis=1)

kl_div_sfvq_mean = np.mean(kl_div_array_sfvq, axis=1)
std_sfvq_mean = np.mean(std_array_sfvq, axis=1)
worst_case_disc_sfvq_mean = np.mean(worst_case_disclosure_array_sfvq, axis=1)
sparseness_sfvq_mean = np.mean(sparseness_array_sfvq, axis=1)
avg_disc_sfvq_mean = np.mean(average_disclosure_sfvq, axis=1)


kl_div_vq_quan_low = np.quantile(kl_div_array_vq,.025, axis=1)
kl_div_vq_quan_high = np.quantile(kl_div_array_vq,.975, axis=1)
std_vq_quan_low = np.quantile(std_array_vq,.025, axis=1)
std_vq_quan_high = np.quantile(std_array_vq,.975, axis=1)
worst_case_disc_vq_quan_low = np.quantile(worst_case_disclosure_array_vq,.025, axis=1)
worst_case_disc_vq_quan_high = np.quantile(worst_case_disclosure_array_vq,.975, axis=1)
sparseness_vq_quan_low = np.quantile(sparseness_array_vq,.025, axis=1)
sparseness_vq_quan_high = np.quantile(sparseness_array_vq,.975, axis=1)
avg_disc_vq_quan_low = np.quantile(average_disclosure_vq,.025, axis=1)
avg_disc_vq_quan_high = np.quantile(average_disclosure_vq,.975, axis=1)


kl_div_sfvq_quan_low = np.quantile(kl_div_array_sfvq,.025, axis=1)
kl_div_sfvq_quan_high = np.quantile(kl_div_array_sfvq,.975, axis=1)
std_sfvq_quan_low = np.quantile(std_array_sfvq,.025, axis=1)
std_sfvq_quan_high = np.quantile(std_array_sfvq,.975, axis=1)
worst_case_disc_sfvq_quan_low = np.quantile(worst_case_disclosure_array_sfvq,.025, axis=1)
worst_case_disc_sfvq_quan_high = np.quantile(worst_case_disclosure_array_sfvq,.975, axis=1)
sparseness_sfvq_quan_low = np.quantile(sparseness_array_sfvq,.025, axis=1)
sparseness_sfvq_quan_high = np.quantile(sparseness_array_sfvq,.975, axis=1)
avg_disc_sfvq_quan_low = np.quantile(average_disclosure_sfvq,.025, axis=1)
avg_disc_sfvq_quan_high = np.quantile(average_disclosure_sfvq,.975, axis=1)

x_range = np.arange(kl_div_sfvq_mean.shape[0])

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=figsize)
ax1.semilogy(x_range, kl_div_vq_mean, 'red')
ax1.semilogy(x_range, kl_div_sfvq_mean, 'blue')
ax1.fill_between(x_range, kl_div_vq_quan_low, kl_div_vq_quan_high, alpha=0.3, color='red')
ax1.fill_between(x_range, kl_div_sfvq_quan_low, kl_div_sfvq_quan_high, alpha=0.3, color='deepskyblue')
ax1.set_xlabel('Quantization Bitrate')
ax1.set_ylabel('KL Divergence')
ax1.set_xticks(x_range)
ax1.set_xticklabels(bitrate_list)
ax1.set_title('(a)')
leg = ax1.legend(('VQ','SFVQ'), prop={"size": text_size})
for legobj in leg.legendHandles:
    legobj.set_linewidth(legend_size)


ax2.semilogy(x_range, worst_case_disc_vq_mean, 'red')
ax2.semilogy(x_range, worst_case_disc_sfvq_mean, 'blue')
ax2.semilogy(x_range, avg_disc_vq_mean, 'darkred', linestyle='dotted')
ax2.semilogy(x_range, avg_disc_sfvq_mean, 'darkblue', linestyle='dashdot')
ax2.semilogy(x_range, np.asarray(bitrate_list), 'dimgray', linestyle=(0, (5,1)))
ax2.fill_between(x_range, worst_case_disc_vq_quan_low, worst_case_disc_vq_quan_high, alpha=0.3, color='red')
ax2.fill_between(x_range, worst_case_disc_sfvq_quan_low, worst_case_disc_sfvq_quan_high, alpha=0.3, color='deepskyblue')
ax2.fill_between(x_range, avg_disc_vq_quan_low, avg_disc_vq_quan_high, alpha=0.3, color='red')
ax2.fill_between(x_range, avg_disc_sfvq_quan_low, avg_disc_sfvq_quan_high, alpha=0.3, color='deepskyblue')
ax2.set_xlabel('Quantization Bitrate')
ax2.set_ylabel('Disclosure (bits)')
ax2.set_xticks(x_range)
ax2.set_xticklabels(bitrate_list)
ax2.set_title('(b)')
legend_list = ['VQ worst-case disclosure','SFVQ worst-case disclosure', 'VQ average disclosure','SFVQ average disclosure', 'Average disclosure upper bound']
leg = ax2.legend((legend_list), loc=4, bbox_to_anchor=(1.01, -0.015), fontsize=15, framealpha=0.95)
for legobj in leg.legendHandles:
    legobj.set_linewidth(legend_size)


ax3.plot(x_range, sparseness_vq_mean, 'red')
ax3.plot(x_range, sparseness_sfvq_mean, 'blue')
ax3.fill_between(x_range, sparseness_vq_quan_low, sparseness_vq_quan_high, alpha=0.3, color='red')
ax3.fill_between(x_range, sparseness_sfvq_quan_low, sparseness_sfvq_quan_high, alpha=0.3, color='deepskyblue')
ax3.set_xlabel('Quantization Bitrate')
ax3.set_ylabel('Sparseness = $\ell_2 / \ell_1$')
ax3.set_xticks(x_range)
ax3.set_xticklabels(bitrate_list)
ax3.set_title('(c)')
leg = ax3.legend(('VQ','SFVQ'), prop={"size": text_size})
for legobj in leg.legendHandles:
    legobj.set_linewidth(legend_size)


ax4.semilogy(x_range, std_vq_mean, 'red')
ax4.semilogy(x_range, std_sfvq_mean, 'blue')
ax4.fill_between(x_range, std_vq_quan_low, std_vq_quan_high, alpha=0.3, color='red')
ax4.fill_between(x_range, std_sfvq_quan_low, std_sfvq_quan_high, alpha=0.3, color='deepskyblue')
ax4.set_xlabel('Quantization Bitrate')
ax4.set_ylabel('Standard Deviation')
ax4.set_xticks(x_range)
ax4.set_xticklabels(bitrate_list)
ax4.set_title('(d)')
leg = ax4.legend(('VQ','SFVQ'), prop={"size": text_size})
for legobj in leg.legendHandles:
    legobj.set_linewidth(legend_size)

plt.tight_layout()
plt.show()
# plt.savefig('metrics.pdf')

