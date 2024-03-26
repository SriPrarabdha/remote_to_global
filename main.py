import torch
import torch.nn as nn
import torch.distributions as td
from stochman.manifold import EmbeddedManifold
from stochman import nnj
from sklearn.cluster import KMeans
import numpy as np
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class VAE(nn.Module, EmbeddedManifold):
    def __init__(self, layers, batch_size, device, sigma=1e-6, sigma_z=0.1):
        super(VAE, self).__init__()
        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers
        self.num_clusters = 500  # Number of clusters in the RBF k_mean
        self.dec_std_pos = None
        self.device = device
        self.kl_coeff = 1.0

        # Encoder
        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.BatchNorm1d(in_features))
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features), nnj.Softplus()))
        enc.append(nnj.Linear(out_features, self.d))
        self.encoder_loc = nnj.Sequential(*enc)

        # Decoder
        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.BatchNorm1d(in_features))
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features), nnj.Softplus()))
        dec.pop(0)
        dec.append(nnj.Linear(out_features, self.p))
        self.decoder_loc = nnj.Sequential(*dec)

        self.encoder_scale_fixed = nn.Parameter(torch.tensor([sigma_z]), requires_grad=False)
        self.decoder_scale_pos = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.prior_loc = nn.Parameter(torch.zeros(self.d), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

    def embed(self, points, jacobian=False):
        std_scale = 1.0
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)
        mu_pos = self.decode(points, train_rbf=True, jacobian=False)[0]
        std = self.dec_std_pos(points, jacobian=False)
        embedded = torch.cat((mu_pos.mean, std_scale * std), dim=2)
        if not is_batched:
            embedded = embedded.squeeze(0)
        return embedded

    def encode(self, x, train_rbf=False):
        z_loc = self.encoder_loc(x)
        z_scale = self.encoder_scale_fixed
        z_distribution = td.Independent(td.Normal(loc=z_loc, scale=z_scale, validate_args=False), 1), z_loc
        return z_distribution

    def decode(self, z, train_rbf=False, jacobian=False):
        x_loc = self.decoder_loc(z.view(-1, self.d))
        position_scale = self.decoder_scale_pos + 1e-10
        position_loc = x_loc
        x_shape = list(z.shape)
        x_shape[-1] = position_loc.shape[-1]
        position_distribution = td.Independent(td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=position_scale), 1)
        return position_distribution

    def init_std(self, x, load_clusters=False):
        x = x.to("cuda")
        self.train_var = True
        with torch.no_grad():
            _, z = self.encode(x, train_rbf=True)
        d = z.shape[1]
        inv_max_std = np.sqrt(1e-12)
        beta = 10.0 / z.std(dim=0).mean()
        rbf_beta = beta * torch.ones(1, self.num_clusters, device="cuda")
        k_means = KMeans(n_clusters=self.num_clusters).fit(z.cpu().numpy())
        if load_clusters:
            k_means = pickle.load(open("../Clusters/" + "clusters.p", "rb"))
        
        centers = torch.tensor(k_means.cluster_centers_).to("cuda")
        self.dec_std_pos = nnj.Sequential(
            nnj.RBF(d, self.num_clusters, points=centers, beta=rbf_beta),
            nnj.PosLinear(self.num_clusters, 1, bias=False),
            nnj.Reciprocal(inv_max_std),
            nnj.PosLinear(1, self.p)
        )
        self.dec_std_pos.to(self.device)
        cluster_centers = k_means.cluster_centers_
        return cluster_centers

    def fit_std(self, data_loader, num_epochs, model):
        params = list(self.dec_std_pos.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        for epoch in range(num_epochs):
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                loss, loss_kl, loss_log = self.loss_function_elbo(data, train_rbf=True, n_samples=100)
                loss.backward()
                optimizer.step()
            print('Training RBF Networks ====> Epoch: {}/{}'.format(epoch, num_epochs))

    def loss_function_elbo(self, x, train_rbf, n_samples):
        q, _ = self.encode(x, train_rbf=train_rbf)
        z = q.rsample(torch.Size([n_samples]))
        px_z = self.decode(z, train_rbf=train_rbf)
        log_p = torch.mean(px_z.log_prob(x), dim=1)
        kl = -0.5 * torch.sum(1 + q.variance.log() - q.mean.pow(2) - q.variance) * self.kl_coeff
        elbo = torch.mean(log_p - kl, dim=0)
        log_mean = torch.mean(log_p, dim=0)
        return -elbo, kl, log_mean

    def disable_training(self):
        for module in self.encoder_loc._modules.values():
            module.training = False
        for module in self.decoder_loc._modules.values():
            module.training = False

def train_model(model, train_loader, test_loader, num_epochs_vae, num_epochs_rbf, device, n_samples, learning_rate_vae, learning_rate_rbf):
    # Regularization-focused training of the VAE
    model.activate_KL = True
    vae_params = list(model.encoder_loc.parameters()) + list(model.decoder_loc.parameters())
    vae_optimizer = optim.Adam(vae_params, lr=learning_rate_vae)
    print("Starting Regularization focused training")
    for epoch in range(num_epochs_vae):
        model.train()
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            vae_optimizer.zero_grad()
            loss, loss_kl, loss_log = model.loss_function_elbo(data, train_rbf=False, n_samples=n_samples)
            loss.backward()
            vae_optimizer.step()
        print(f'Regularization Training - Epoch [{epoch+1}/{num_epochs_vae}], VAE Loss: {loss.item():.4f}, KL Loss: {loss_kl.item():.4f}, Log Likelihood: {loss_log.item():.4f}')

    # Reconstruction-focused training of the VAE
    model.activate_KL = False
    model.kl_coeff = 0.1
    vae_params = list(model.decoder_loc.parameters())
    vae_optimizer = optim.Adam(vae_params, lr=learning_rate_vae)
    print("Starting Reconstruction focused training")
    for epoch in range(num_epochs_vae):
        if epoch == num_epochs_vae // 2:
            model.empowered_quaternions = True
        model.train()
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            vae_optimizer.zero_grad()
            loss, loss_kl, loss_log = model.loss_function_elbo(data, train_rbf=False, n_samples=n_samples)
            loss.backward()
            vae_optimizer.step()
        print(f'Reconstruction Training - Epoch [{epoch+1}/{num_epochs_vae}], VAE Loss: {loss.item():.4f}, KL Loss: {loss_kl.item():.4f}, Log Likelihood: {loss_log.item():.4f}')
    model.empowered_quaternions = False

    # Training of the RBF/Variance networks
    model.init_std(train_data.float(), load_clusters=False)
    rbf_params = list(model.dec_std_pos.parameters())
    rbf_optimizer = optim.Adam(rbf_params, lr=learning_rate_rbf)
    print("Starting RBF Training")

    for epoch in range(num_epochs_rbf):
        model.train_var = True
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            rbf_optimizer.zero_grad()
            loss, loss_kl, loss_log = model.loss_function_elbo(data, train_rbf=True, n_samples=n_samples)
            loss.backward()
            rbf_optimizer.step()
        print(f'RBF Training - Epoch [{epoch+1}/{num_epochs_rbf}], VAE Loss: {loss.item():.4f}, KL Loss: {loss_kl.item():.4f}, Log Likelihood: {loss_log.item():.4f}')

    # Compute and store the latent representation of the training data
    model.eval()
    with torch.no_grad():
        latent_representation = []
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            q, _ = model.encode(data, train_rbf=True)
            latent_representation.append(q.mean.detach().cpu().numpy())
        latent_representation = np.concatenate(latent_representation, axis=0)
        np.save('latent_representation.npy', latent_representation)

    return model

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dof = 1000
encoder_scales = [1.0]
n_samples = 100
num_epochs_rbf = 1000
num_epochs_vae = 1000
batch_size = 256

learning_rate_vae = 1e-3
learning_rate_rbf = 1e-3
data = np.load('Paul_poincare.npy')

# Convert the numpy array to a tensor
train_data = torch.from_numpy(data).float()

# Create a TensorDataset using the train data
train_dataset = TensorDataset(train_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = train_loader

model = VAE(layers=[dof, 512, 128, 32,2], batch_size=batch_size, device=device, sigma_z=encoder_scales[0]).to(device)
#model.init_std(train_data.tensors[0].float(), load_clusters=False)
trained_model = train_model(model, train_loader, test_loader, num_epochs_vae,num_epochs_rbf, device, n_samples, learning_rate_vae, learning_rate_rbf)
