import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from plain_vq import Plain_VQ
from matplotlib.backends.backend_pdf import PdfPages
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.random.randn(int(2**17), 64).astype(np.float32)
data = torch.from_numpy(data)
num_data_samples = data.shape[0]
data_dim = data.shape[1]

# Training Hyper-Parameters
num_epochs = 10
log_interval = 500 # save training logs after "log_interval" number of batches
replacement_num_batches = 500 # num of batches to replace unused (inactive) codebooks with active ones
parser = argparse.ArgumentParser()
parser.add_argument('--desired_bitrate', type=int, default=6) # quantization bitrate
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-3)
args = parser.parse_args()


experiment_name = f'{args.desired_bitrate}bit_bs{args.batch_size}_lr{args.learning_rate}'
base_address = "vq_training_logs" # address to save training logs and outputs
os.makedirs(base_address, exist_ok=True)
pp = PdfPages(os.path.join(base_address,f'hist_{experiment_name}.pdf'))


if num_data_samples % args.batch_size==0:
    num_batches = int(np.floor(num_data_samples/args.batch_size))
else:
    num_batches = int(np.floor(num_data_samples / args.batch_size) + 1)

num_codebooks = int(2**args.desired_bitrate)
num_training_updates = int(num_epochs * num_batches)
num_logs = num_epochs * int(np.floor(num_batches/log_interval))

milestones = [int(0.6*num_epochs), int(0.8*num_epochs)] # learning rate scheduler

# initialization (define initial codebooks)
temp = np.random.permutation(num_data_samples)
initial_codebooks =data[temp][0:num_codebooks].to(device)
vector_quantizer = Plain_VQ(initial_codebooks, device=device)
vector_quantizer.to(device)

def plot_function(hist, epoch_number):
    num_bars = np.size(hist)
    fig = plt.figure(figsize=(10,6))
    plt.subplot(1,1,1)
    plt.bar(np.arange(1, num_bars + 1), height=hist, width=1)
    plt.title(f'Original Hist | Epoch No = {epoch_number}' )
    return fig

optimizer = optim.Adam(vector_quantizer.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

# arrays to save the training logs
total_vq_loss = np.zeros((num_logs,))
total_perplexity = np.zeros((num_logs,))
used_cb_indices = np.zeros((num_codebooks,), dtype=np.float32)

# define counters to use them during training
global_batch_counter = 0
logging_epoch = 0

for epoch in range(num_epochs):
    print('######################### epoch =', epoch + 1, '#########################')

    vq_loss_accumulator = 0
    perplexity_accumulator = 0

    batch_counter = 0
    small_batch_counter = 0
    input_data = data[torch.randperm(num_data_samples), :] # shuffle the data

    for j in range(num_batches):

        if j == (num_batches - 1):
            data_batch = input_data[j * args.batch_size:].to(device)
        else:
            data_batch = input_data[j * args.batch_size: (j + 1) * args.batch_size].to(device)

        optimizer.zero_grad()

        quantized, vq_loss, perplexity, min_indiecs = vector_quantizer(data_batch)

        loss = vq_loss

        loss.backward()
        optimizer.step()

        used_cb_indices[min_indiecs.astype(np.int64)] += 1

        global_batch_counter += 1
        small_batch_counter += 1
        batch_counter += 1

        vq_loss_accumulator += vq_loss.item()
        perplexity_accumulator += perplexity.item()

        vq_loss_average = vq_loss_accumulator / small_batch_counter
        perplexity_average = perplexity_accumulator / small_batch_counter

        # replace unused codebooks
        if (global_batch_counter % replacement_num_batches == 0) and (0 < global_batch_counter <= num_training_updates - (4*replacement_num_batches)):
            vector_quantizer.replace_unused_codebooks(replacement_num_batches)

        # print logs
        if batch_counter % log_interval == 0:
            total_vq_loss[logging_epoch] = vq_loss_average
            total_perplexity[logging_epoch] = perplexity_average

            print("VQ Loss:{:.6f}, Perpexlity:{:.4f}".format(vq_loss_average, perplexity_average))
            print(f'Batch No: {batch_counter}/{num_batches} | Loss: {vq_loss_average:.6f} | Perp: {perplexity_average:.4f}')

            vq_loss_accumulator = perplexity_accumulator = 0
            small_batch_counter = 0
            logging_epoch += 1

    scheduler.step()

    fig = plot_function(np.log10(used_cb_indices + 1), epoch + 1)
    pp.savefig(fig, bbox_inches='tight')

# save training outputs
np.save(os.path.join(base_address,f"total_vq_loss_{experiment_name}.npy"), total_vq_loss)
np.save(os.path.join(base_address,f"total_perplexity_{experiment_name}.npy"), total_perplexity)
np.save(os.path.join(base_address,f"codebook_{experiment_name}.npy"), vector_quantizer.codebooks.cpu().detach().numpy())

print('******** optimization finished ********')

#################### Inference Phase ####################
quantized_data = torch.zeros_like(data)

eval_batch_size = 64
if num_data_samples % eval_batch_size==0:
    num_batches = int(np.floor(num_data_samples/eval_batch_size))
else:
    num_batches = int(np.floor(num_data_samples / eval_batch_size) + 1)

optimized_cb = vector_quantizer.codebooks.detach().clone()

for i in range(num_batches):
    if i == (num_batches-1):
        data_batch = data[(i * eval_batch_size): ].to(device)
        distance = (data_batch.unsqueeze(-1) - optimized_cb.t().unsqueeze(0)).square().sum(dim=1)
        integer_index = distance.argmin(dim=1)
        quantized_batch = optimized_cb[integer_index]
        quantized_data[(i * eval_batch_size):] = quantized_batch
    else:
        data_batch = data[(i*eval_batch_size):((i+1)*eval_batch_size)].to(device)
        distance = (data_batch.unsqueeze(-1) - optimized_cb.t().unsqueeze(0)).square().sum(dim=1)
        integer_index = distance.argmin(dim=1)
        quantized_batch = optimized_cb[integer_index]
        quantized_data[(i * eval_batch_size):((i + 1) * eval_batch_size)] = quantized_batch


mse = torch.mean(torch.square(data - quantized_data))

fig = plt.figure(figsize=(15, 5))
total_vq_loss = total_vq_loss.reshape(-1,1)
plt.plot(total_vq_loss)
plt.title(f'VQ Loss | final MSE = {mse:.4f}')
pp.savefig(fig, bbox_inches='tight')

fig = plt.figure(figsize=(15, 5))
total_perplexity = total_perplexity.reshape(-1,1)
plt.plot(total_perplexity)
plt.title('Perplexity')
pp.savefig(fig, bbox_inches='tight')

pp.close()
print("End of the code execution...!!!")
