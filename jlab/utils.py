import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import distance
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    def __init__(
        self,
        num_features: int = 32,
        hidden_dim: list = [32, 16],
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        """Simple MLP classifier class"""
        super(MLPClassifier, self).__init__()

        layers = []
        current_dim = num_features

        # hidden layers
        for h_dim in hidden_dim:
            layers.append(nn.Linear(current_dim, h_dim, bias=True))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            current_dim = h_dim

        # output layer
        layers.append(nn.Linear(current_dim, output_dim, bias=True))

        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_sigmoid=True):
        log_r = self.mlp(x)
        if use_sigmoid:
            return self.sigmoid(log_r)
        return log_r


def train_gan_model(
    model_G: nn.Module,
    model_D: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    latent_size: int,
    learning_rate: float,
    num_epochs: int,
    device: torch.device,
    model_path: str = "./outputs/best_model_G.pth",
    instance_noise_std: float = 0.0,
    label_smoothing: float = 0.0,
    g_use_sigmoid: bool = True,
):
    model_G.float().to(device)
    model_D.float().to(device)
    optimizer_G = optim.Adam(model_G.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(model_D.parameters(), lr=learning_rate)
    # scheduler_G = ReduceLROnPlateau(optimizer_G, mode="min", factor=0.5, patience=10)
    # scheduler_D = ReduceLROnPlateau(optimizer_D, mode="min", factor=0.5, patience=10)
    criterion = nn.BCELoss()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def apply_noise(x, std):
        if std > 0:
            return x + torch.randn_like(x) * std
        return x

    def train_one_epoch(model_D, model_G, data_loader, device):
        total_train_loss_D = 0
        total_train_loss_G = 0
        model_D.train()
        model_G.train()
        for (x_batch,) in data_loader:
            b_size = x_batch.shape[0]
            real_target = 1.0 - label_smoothing
            fake_target = 0.0

            label_real = (
                torch.full((b_size,), real_target).float().to(device, non_blocking=True)
            )
            x_batch = x_batch.float().to(device, non_blocking=True)
            optimizer_D.zero_grad(set_to_none=True)

            y_batch_real = model_D(apply_noise(x_batch, instance_noise_std))
            loss_D_real = criterion(y_batch_real.view(-1), label_real)
            loss_D_real.backward()

            noise = (
                torch.randn(b_size, latent_size).float().to(device, non_blocking=True)
            )
            label_fake = (
                torch.full((b_size,), fake_target).float().to(device, non_blocking=True)
            )

            x_fake = model_G(noise, use_sigmoid=g_use_sigmoid)
            y_batch_fake = model_D(apply_noise(x_fake.detach(), instance_noise_std))
            loss_D_fake = criterion(y_batch_fake.view(-1), label_fake)
            loss_D_fake.backward()

            optimizer_D.step()

            optimizer_G.zero_grad(set_to_none=True)
            label_gen = torch.ones(b_size).float().to(device, non_blocking=True)
            y_batch_fake_G = model_D(apply_noise(x_fake, instance_noise_std))
            loss_G = criterion(y_batch_fake_G.view(-1), label_gen)
            loss_G.backward()
            optimizer_G.step()

            total_train_loss_D += loss_D_real.item() + loss_D_fake.item()
            total_train_loss_G += loss_G.item()
        return (
            total_train_loss_G / len(data_loader),
            total_train_loss_D / len(data_loader),
        )

    @torch.no_grad()
    def val_one_epoch(model_D, model_G, data_loader, device):
        total_val_loss_D = 0
        total_val_loss_G = 0
        model_D.eval()
        model_G.eval()
        for (x_batch,) in data_loader:
            b_size = x_batch.shape[0]
            label_real = torch.ones(b_size).float().to(device, non_blocking=True)
            x_batch = x_batch.float().to(device, non_blocking=True)

            y_batch_real = model_D(x_batch)
            loss_D_real = criterion(y_batch_real.view(-1), label_real)

            noise = (
                torch.randn(b_size, latent_size).float().to(device, non_blocking=True)
            )
            label_fake = torch.zeros(b_size).float().to(device, non_blocking=True)

            x_fake = model_G(noise, use_sigmoid=g_use_sigmoid)
            y_batch_fake = model_D(x_fake)
            loss_D_fake = criterion(y_batch_fake.view(-1), label_fake)

            y_batch_fake_G = model_D(x_fake)
            loss_G = criterion(y_batch_fake_G.view(-1), label_real)

            total_val_loss_D += loss_D_real.item() + loss_D_fake.item()
            total_val_loss_G += loss_G.item()
        return (
            total_val_loss_G / len(data_loader),
            total_val_loss_D / len(data_loader),
        )

    train_history_G, train_history_D = [], []
    val_history_G, val_history_D = [], []
    best_val_loss_G = 0.0

    for epoch in range(num_epochs):
        avg_train_loss_G, avg_train_loss_D = train_one_epoch(
            model_D, model_G, train_loader, device
        )
        avg_val_loss_G, avg_val_loss_D = val_one_epoch(
            model_D, model_G, val_loader, device
        )

        # scheduler_G.step(avg_val_loss_G)
        # scheduler_D.step(avg_val_loss_D)

        print(
            f"\t> Epoch {epoch + 1:03d} | Train Loss G: {avg_train_loss_G:.4f} D: {avg_train_loss_D:.4f} | Val Loss G: {avg_val_loss_G:.4f} D: {avg_val_loss_D:.4f}"
        )

        if avg_val_loss_G > best_val_loss_G:
            best_val_loss_G = avg_val_loss_G
            torch.save(model_G.state_dict(), model_path)

        train_history_G.append(avg_train_loss_G)
        train_history_D.append(avg_train_loss_D)
        val_history_G.append(avg_val_loss_G)
        val_history_D.append(avg_val_loss_D)

    if os.path.exists(model_path):
        model_G.load_state_dict(torch.load(model_path))
    return (
        model_G,
        np.array(train_history_G),
        np.array(train_history_D),
        np.array(val_history_G),
        np.array(val_history_D),
    )


def calculate_distribution_metrics(real_data, gen_data):
    """Calculates Kolmogorov-Smirnov test and Jensen-Shannon divergence."""
    ks_stat, ks_p = ks_2samp(real_data.flatten(), gen_data.flatten())
    bins = np.linspace(
        min(real_data.min(), gen_data.min()), max(real_data.max(), gen_data.max()), 100
    )
    p, _ = np.histogram(real_data, bins=bins, density=True)
    q, _ = np.histogram(gen_data, bins=bins, density=True)
    p = p + 1e-10
    q = q + 1e-10
    p /= p.sum()
    q /= q.sum()
    js_div = distance.jensenshannon(p, q)
    return {"ks_stat": ks_stat, "ks_p_value": ks_p, "js_divergence": js_div}


def prepare_data(file_path: str):
    """Loads age data from .npy file and scales it using StandardScaler."""
    data = np.load(file_path)
    age_values = data["age"].astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    age_scaled = scaler.fit_transform(age_values)

    return age_values, age_scaled, scaler


def plot_comparison_with_pull(
    real_data, gen_data, bins, title, save_path, label_gen="Generated", gen_weights=None
):
    """
    Plots a comparison histogram with a pull plot below.
    Pull = (Real Count - Gen Count) / sqrt(Real Count)
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    # Calculate counts and bin centers
    real_counts, bin_edges = np.histogram(real_data, bins=bins)
    gen_counts, _ = np.histogram(gen_data, bins=bins, weights=gen_weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Errors for real data (sqrt(N))
    real_errors = np.sqrt(real_counts)

    # Top plot: Normalized Histograms
    norm_real = real_counts / (np.sum(real_counts) * np.diff(bin_edges))
    norm_gen = gen_counts / (np.sum(gen_counts) * np.diff(bin_edges))
    norm_errors = real_errors / (np.sum(real_counts) * np.diff(bin_edges))

    ax1.errorbar(
        bin_centers,
        norm_real,
        yerr=norm_errors,
        fmt="o",
        color="black",
        label="True Distribution",
        markersize=5,
    )
    ax1.hist(
        bin_centers,
        bins=bin_edges,
        weights=norm_gen,
        histtype="step",
        color="red",
        label=label_gen,
        linewidth=2,
    )
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False)
    ax1.set_title(title)

    # Bottom plot: Pull Plot
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = (norm_real - norm_gen) / norm_errors
        pulls[~np.isfinite(pulls)] = 0

    ax2.errorbar(
        bin_centers,
        pulls,
        yerr=np.ones_like(pulls),
        fmt="o",
        color="black",
        markersize=5,
    )
    ax2.axhline(0, color="blue", linestyle="--")
    ax2.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(-1, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Pull")
    ax2.set_xlabel("Age")
    ax2.set_ylim(-5, 5)

    # plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


class MLPVAE(nn.Module):
    def __init__(
        self,
        num_features: int = 1,
        hidden_dim: list = [32, 32],
        latent_dim: int = 2,
    ):
        super(MLPVAE, self).__init__()
        encoder_layers = []
        current_dim = num_features
        for h_dim in hidden_dim:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)

        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dim):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, num_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train_vae_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    num_epochs: int,
    device: torch.device,
    model_path: str = "./outputs/best_model_vae.pth",
):
    model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def train_one_epoch(model, data_loader, optimizer, device):
        model.train()
        total_loss = 0
        for (x_batch,) in data_loader:
            x_batch = x_batch.float().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch)
            loss = vae_loss_function(recon_batch, x_batch, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        return total_loss / len(data_loader.dataset)

    @torch.no_grad()
    def val_one_epoch(model, data_loader, device):
        model.eval()
        total_loss = 0
        for (x_batch,) in data_loader:
            x_batch = x_batch.float().to(device)
            recon_batch, mu, logvar = model(x_batch)
            loss = vae_loss_function(recon_batch, x_batch, mu, logvar)
            total_loss += loss.item()
        return total_loss / len(data_loader.dataset)

    train_history, val_history = [], []
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device)
        avg_val_loss = val_one_epoch(model, val_loader, device)
        scheduler.step(avg_val_loss)
        print(
            f"\t> Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model, np.array(train_history), np.array(val_history)
