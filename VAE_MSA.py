import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


activation_dict = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
}


class VAE(pl.LightningModule):
    def __init__(self, input_length, num_classes, device="cpu", h_params=None):
        super().__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self._device = device
        self.h_params = {
            "z_dim": 2,
            "lr": 1e-3,
            "dropout": 0.1,
            "beta": 0.01,
            "activation": "LeakyReLU",
            "dim1": 1500,
            "dim2": 1000,
            "is_annealing": True,
            "max_epochs": 20,
        }
        if h_params is not None:
            self.h_params.update(h_params)
        self.save_hyperparameters(self.h_params)

        encoder_dims = [self.h_params["dim1"], self.h_params["dim2"]]
        decoder_dims = [self.h_params["dim2"], self.h_params["dim1"]]
        z_dim = self.h_params["z_dim"]
        activation = activation_dict[self.h_params["activation"]]

        # Build Encoder
        modules = []
        input_dim = self.input_length * self.num_classes
        in_dim = input_dim
        for h_dim in encoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.Dropout(self.h_params["dropout"]),
                    nn.BatchNorm1d(h_dim),
                    activation(),
                )
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_mu = nn.Linear(encoder_dims[-1], z_dim)
        self.encoder_log_var = nn.Linear(encoder_dims[-1], z_dim)

        # Build Decoder
        modules = []
        in_dim = z_dim
        for h_dim in decoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.Dropout(self.h_params["dropout"]),
                    nn.BatchNorm1d(h_dim),
                    activation(),
                )
            )
            in_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(decoder_dims[-1], input_dim)))
        self.decoder = nn.Sequential(*modules)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.eval_loss = []

    def encode(self, x):  # x: [N, L]
        x_one_hot = F.one_hot(x.long(), self.num_classes).float()  # [N, L, C]
        x_one_hot_flat = x_one_hot.reshape(x_one_hot.shape[0], -1)  # [N, L*C]
        h = self.encoder(x_one_hot_flat)
        mu = self.encoder_mu(h)
        log_var = self.encoder_log_var(h)
        return mu, log_var

    def decode(self, z):
        h = self.decoder(z)  # [N, L*C]
        N = h.shape[0]
        L = self.input_length
        C = self.num_classes
        h_reshaped = h.view(N, L, C)  # [N, L, C]
        h_softmax = F.softmax(h_reshaped, dim=2)
        h_final = h_softmax.view(N, L * C)  # [N, L*C]
        return h_final

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, n_samples=1):
        mu, log_var = self.encode(x)
        z_samples = self.reparameterize(mu, log_var)
        x_recon = self.decode(z_samples)
        return x_recon, mu, log_var

    def _step(self, batch, batch_idx, is_training=True):
        x = batch.long().to(self._device)  # [N, L]
        x_recon, mu, log_var = self(x)  # x_recon: [N, L*C]
        N, L = x.shape
        C = self.num_classes
        x_recon = x_recon.view(N, L, C)
        x = x.view(-1)  # [N*L]
        x_recon = x_recon.view(-1, C)  # [N*L, C]
        x_recon_logigts = torch.log(x_recon + 1e-6)
        recon_loss = self.loss_fn(x_recon_logigts, x)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        if is_training and self.hparams["is_annealing"]:
            beta = self.h_params["beta"] * self.current_epoch / self.h_params["max_epochs"]
        else:
            beta = self.h_params["beta"]
        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self._step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss},
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self._step(batch, batch_idx, is_training=False)
        self.log_dict({"val_loss": loss}, on_step=False, on_epoch=True)
        self.eval_loss.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self._step(batch, batch_idx, is_training=False)
        self.log_dict({"test_loss": loss}, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.eval_loss.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.h_params["lr"])


def test():
    N = 10
    L = 50
    C = 21
    x = torch.randint(low=0, high=C, size=(N, L))
    test_model = VAE(input_length=L, num_classes=C)
    x_recon, mu, log_var = test_model(x)
    print(f"N: {N}, L: {L}, C: {C}")
    print(f"x: {x.shape}")
    print(f"x_recon: {x_recon.shape}")
    print(f"mu: {mu.shape}")
    print(f"log_var: {log_var.shape}")


if __name__ == "__main__":
    test()
