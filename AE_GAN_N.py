#!/usr/bin/env python3
# ==========================================================
#  Auto-Encoding GAN for tabular data   (dimension d ≥ 2)
#  PCA is fitted ONCE on the real set; re-used every snapshot
# ==========================================================
import argparse, os, random, itertools
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------- CLI ---------------------------------
P = argparse.ArgumentParser()
P.add_argument("--dataset_name", type=str,  default="R15")
P.add_argument("--data_dim",    type=int,   default=2)
P.add_argument("--n_classes",   type=int,   default=15)
P.add_argument("--latent_dim",  type=int,   default=100)
P.add_argument("--n_epochs",    type=int,   default=500)
P.add_argument("--batch_size",  type=int,   default=128)
P.add_argument("--lr",          type=float, default=1e-3)
P.add_argument("--b1",          type=float, default=0.5)
P.add_argument("--b2",          type=float, default=0.999)
P.add_argument("--sample_interval", type=int, default=10)
opt = P.parse_args(); print(opt)

# -------------------- Device ------------------------------
device = (torch.device("mps")   if torch.backends.mps.is_available() else
          torch.device("cuda")  if torch.cuda.is_available()         else
          torch.device("cpu"))

# -------------------- Paths -------------------------------
root   = os.getcwd()
outdir = os.path.join(root, f"{opt.dataset_name}_images")
for sub in ["test_result", "classifier_result",
            "generator_model", "discriminator_model"]:
    os.makedirs(os.path.join(outdir, sub), exist_ok=True)

# -------------------- Helpers -----------------------------
def w_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def to_onehot(y, n):
    o = torch.zeros(len(y), n, device=device)
    o[torch.arange(len(y)), y] = 1.
    return o

def normalise(x, lo=-0.8, hi=0.8):
    mn, mx = x.min(0, keepdims=True)[0], x.max(0, keepdims=True)[0]
    span   = (mx-mn).clamp_min(1e-7)
    return (x-mn)/span*(hi-lo)+lo

# -------------------- Networks ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        d = opt.data_dim
        self.net = nn.Sequential(
            nn.Linear(opt.latent_dim+opt.n_classes, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,  64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear( 64, d), nn.Tanh()
        )
    def forward(self, z, y):
        return self.net(torch.cat([z, y], 1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        d = opt.data_dim
        self.feat = nn.Sequential(
            nn.Linear(d,  64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128,256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256,512), nn.ReLU()
        )
        self.adv = nn.Sequential(nn.Linear(512,1),            nn.Sigmoid())
        self.aux = nn.Sequential(nn.Linear(512,opt.n_classes), nn.Softmax(1))
    def forward(self,x):
        f = self.feat(x)
        return self.adv(f), self.aux(f), f

G, D = (Generator().to(device).apply(w_init),
        Discriminator().to(device).apply(w_init))

adv_loss  = nn.BCELoss()
cls_loss  = nn.CrossEntropyLoss()
opt_G     = torch.optim.Adam(G.parameters(),                    lr=opt.lr, betas=(opt.b1,opt.b2))
opt_D     = torch.optim.Adam(D.parameters(),                    lr=opt.lr, betas=(opt.b1,opt.b2))
opt_Info  = torch.optim.Adam(itertools.chain(G.parameters(),
                                             D.parameters()),  lr=opt.lr, betas=(opt.b1,opt.b2))

# -------------------- Data --------------------------------
data_file = os.path.join(root, "dataset", f"{opt.dataset_name}.txt")
X         = np.loadtxt(data_file, dtype=np.float32)[:, :opt.data_dim]
X         = normalise(torch.from_numpy(X))
loader    = DataLoader(TensorDataset(X), batch_size=opt.batch_size,
                       shuffle=True, drop_last=False)

# -------------------- PCA (fit once) ----------------------
pca      = PCA(n_components=2).fit(X.numpy())
real_2d  = pca.transform(X.numpy())        # for plotting

# -------------------- Plot helpers ------------------------
# ──────────────────────────────────────────────────────────
#  Plot helpers  ——  identical style to original code
# ──────────────────────────────────────────────────────────
def plot_pca(gen, epoch):
    gen_2d     = pca.transform(gen.cpu().numpy())
    plt.figure(figsize=(6, 6))
    xmin, xmax = real_2d[:,0].min(), real_2d[:,0].max()
    ymin, ymax = real_2d[:,1].min(), real_2d[:,1].max()
    plt.xlim(xmin-.05, xmax+.05);  plt.ylim(ymin-.05, ymax+.05)

    # real data  ➜  blue points
    plt.scatter(real_2d[:, 0], real_2d[:, 1],
                color='b', marker='.', alpha=1.0)

    # generated data  ➜  class-coloured digits
    for c in range(opt.n_classes):
        pts = gen_2d[c*each:(c+1)*each]
        col = "#" + ''.join(random.choice("123456789ABCDEF") for _ in range(6))
        for x, y in pts:
            plt.text(x, y, str(c), color=col,
                     fontdict={'weight': 'bold', 'size': 9})

    plt.title(f"{opt.dataset_name} - epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "test_result", f"gan_{epoch}.jpg"))
    plt.close()


def plot_classifier(epoch):
    with torch.no_grad():
        _, preds, _ = D(X.to(device))
        labels = preds.argmax(1).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.xlim(-1.0, 1.0);  plt.ylim(-1.0, 1.0)

    for c in range(opt.n_classes):
        col = "#" + ''.join(random.choice("123456789ABCDEF") for _ in range(6))
        pts = real_2d[labels == c]
        for x, y in pts:
            plt.text(x, y, str(c), color=col,
                     fontdict={'weight': 'bold', 'size': 9}, alpha=0.5)

    plt.title(f"{opt.dataset_name} classifier result (epoch={epoch})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "classifier_result",
                             f"classifier_{epoch}.jpeg"))
    plt.close()


# -------------------- Training ----------------------------
each  = 20      # samples per generator for visual snapshots
alpha = 0.0

for epoch in range(opt.n_epochs):

    for (real_batch,) in loader:
        real_batch = real_batch.to(device)
        B          = real_batch.size(0)

        # ==================================================
        #              Adversarial stage
        # ==================================================
        z     = torch.randn(B, opt.latent_dim, device=device)
        y_lab = torch.randint(0, opt.n_classes, (B,), device=device)
        y_oh  = to_onehot(y_lab, opt.n_classes)

        # -------------- Generator -------------------------
        opt_G.zero_grad()
        gen_batch              = G(z, y_oh)
        adv_fake, _, _         = D(gen_batch)
        g_loss                 = adv_loss(adv_fake, torch.ones_like(adv_fake))
        g_loss.backward()
        opt_G.step()

        # -------------- Discriminator ---------------------
        opt_D.zero_grad()
        adv_real, _, _         = D(real_batch)
        adv_fake, _, _         = D(gen_batch.detach())
        d_loss = (adv_loss(adv_real, torch.ones_like(adv_real)) +
                  adv_loss(adv_fake, torch.zeros_like(adv_fake))) * 0.5
        d_loss.backward()
        opt_D.step()

        # ==================================================
        #     Info / clustering stage  (fresh forward pass)
        # ==================================================
        opt_Info.zero_grad()

        # regenerate visual samples for info-loss
        z_vis   = torch.randn(each*opt.n_classes, opt.latent_dim, device=device)
        tgt_vis = torch.arange(opt.n_classes, device=device).repeat_interleave(each)
        oh_vis  = to_onehot(tgt_vis, opt.n_classes)
        gen_vis = G(z_vis, oh_vis)

        _, cls_g,  _      = D(gen_vis)
        _, cls_r,  feat_r = D(real_batch)

        g_cls_loss  = cls_loss(cls_g, tgt_vis)

        # --- K-means on detached (no-grad) normalized features
        feat_r_np   = F.normalize(feat_r, 1).detach().cpu().numpy()
        r_labels_np = KMeans(opt.n_classes, n_init="auto").fit_predict(feat_r_np)
        r_labels    = torch.tensor(r_labels_np, device=device, dtype=torch.long)

        r_cls_loss  = cls_loss(cls_r, r_labels)

        info_loss   = g_cls_loss + alpha * r_cls_loss
        info_loss.backward()
        opt_Info.step()

    # schedule α
    alpha = 0.001 if epoch < 0.7*opt.n_epochs else 0.1
    print(f"[{epoch:4d}/{opt.n_epochs}]  D:{d_loss.item():.4f}  "
          f"G:{g_loss.item():.4f}  Info:{info_loss.item():.4f}")

    # -------- snapshot / plots ----------------------------
    if epoch % opt.sample_interval == 0:
        torch.save(G.state_dict(), os.path.join(outdir,"generator_model",f"gen_{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(outdir,"discriminator_model",f"disc_{epoch}.pth"))
        with torch.no_grad():
            plot_pca(gen_vis, epoch)
            plot_classifier(epoch)

print("✅  Training finished.  Outputs in:", outdir)
