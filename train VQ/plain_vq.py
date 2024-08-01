import torch
import torch.nn.functional as F

class Plain_VQ(torch.nn.Module):
    def __init__(self, initial_codebook, device=torch.device('cuda')):
        super().__init__()

        # initial codebook has the shape of 4 x embedding_dimension: it has 4 codebooks since spacefilli
        self.device = device
        self.embedding_dim = initial_codebook.shape[1]
        self.num_embeddings = initial_codebook.shape[0]
        self.discarding_threshold = 0.01
        self.eps = 1e-12

        self.codebooks = torch.nn.Parameter(initial_codebook, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_embeddings, device=device)


    def forward(self, input_data):

        distances = (input_data.unsqueeze(-1) - self.codebooks.t().unsqueeze(0)).square().sum(dim=1)
        min_indices = distances.argmin(dim=1)
        encodings = F.one_hot(min_indices, self.num_embeddings).float()

        quantized_input = self.codebooks[min_indices]

        # quantized_final = self.nsvq(x, quantized_input).view(input_shape)
        # quantized_final = quantized_input.reshape(input_shape)

        loss = F.mse_loss(input_data, quantized_input)

        with torch.no_grad():
            self.codebooks_used[min_indices] += 1

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_input, loss, perplexity.detach(), min_indices.cpu().detach().numpy()

    def replace_unused_codebooks(self, num_batches):
        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device).clone()

            print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0

