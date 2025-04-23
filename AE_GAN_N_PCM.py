#!/usr/bin/env python3
# ==========================================================
#  Auto-Encoding GAN with PCM clustering  –  any dimension d
#  (plots keep the exact style of your original 2-D code)
# ==========================================================
import argparse, os, random, itertools
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cmeans import pcm                         # ← you provide this
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------- CLI ---------------------------------
p = argparse.ArgumentParser()
p.add_argument("--dataset_name", type=str, default="R15")
p.add_argument("--data_dim",    type=int, default=2)
p.add_argument("--n_classes",   type=int, default=15)
p.add_argument("--latent_dim",  type=int, default=100)
p.add_argument("--n_epochs",    type=int, default=500)
p.add_argument("--batch_size",  type=int, default=128)
p.add_argument("--lr",          type=float, default=4e-4)
p.add_argument("--b1",          type=float, default=0.5)
p.add_argument("--b2",          type=float, default=0.99)
p.add_argument("--sample_interval", type=int, default=10)
# PCM hyper-parameters
p.add_argument("--pcm_m",  type=float, default=2.0)
p.add_argument("--pcm_e",  type=float, default=1e-4)
p.add_argument("--pcm_max_iter", type=int, default=50)
opt = p.parse_args(); print(opt)

# -------------------- Device ------------------------------
device = (torch.device("mps")    if torch.backends.mps.is_available()
          else torch.device("cuda:0") if torch.cuda.is_available()
          else torch.device("cpu"))

# -------------------- Paths -------------------------------
root   = os.getcwd()
outdir = os.path.join(root, f"{opt.dataset_name}_images_PCM")
for sub in ["test_result", "classifier_result",
            "generator_model", "discriminator_model"]:
    os.makedirs(os.path.join(outdir, sub), exist_ok=True)

# -------------------- Helpers -----------------------------
def w_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0)

def to_onehot(y, n):
    h = torch.zeros(len(y), n, device=device)
    h[torch.arange(len(y)), y] = 1.
    return h

def normalise(a, lo=-0.8, hi=0.8):
    mn, mx = a.min(0, keepdims=True)[0], a.max(0, keepdims=True)[0]
    span = (mx-mn).clamp_min(1e-7)
    return (a-mn)/span*(hi-lo)+lo

def rand_colour():                   # hex colour for digits
    return "#" + ''.join(random.choice("123456789ABCDEF") for _ in range(6))

# -------------------- Nets --------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        d = opt.data_dim
        self.net = nn.Sequential(
            nn.Linear(opt.latent_dim+opt.n_classes,256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64,d),    nn.Tanh()
        )
    def forward(self,z,y): return self.net(torch.cat([z,y],1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        d = opt.data_dim
        self.feat = nn.Sequential(
            nn.Linear(d,64),  nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64,128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128,256),nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256,512),nn.ReLU()
        )
        self.adv = nn.Sequential(nn.Linear(512,1), nn.Sigmoid())
        self.aux = nn.Sequential(nn.Linear(512,opt.n_classes), nn.Softmax(1))
    def forward(self,x):
        f = self.feat(x)
        return self.adv(f), self.aux(f), f

G = Generator().to(device).apply(w_init)
D = Discriminator().to(device).apply(w_init)

# -------------------- Loss / Optims -----------------------
adv_loss = nn.BCELoss();   cls_loss = nn.CrossEntropyLoss()
opt_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
opt_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
opt_I = torch.optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                         lr=opt.lr, betas=(opt.b1,opt.b2))

# -------------------- Data --------------------------------
data_file = os.path.join(root,"dataset",f"{opt.dataset_name}.txt")
X = np.loadtxt(data_file, dtype=np.float32)[:, :opt.data_dim]
X = normalise(torch.from_numpy(X))
loader = DataLoader(TensorDataset(X), batch_size=opt.batch_size,
                    shuffle=True, drop_last=False)

# -------------------- PCA (fit once) ----------------------
pca = PCA(n_components=2).fit(X.numpy())
real_2d = pca.transform(X.numpy())                 # cache

# -------------------- Plot helpers ------------------------
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
    plt.figure(figsize=(6,6)); plt.xlim(-1,1); plt.ylim(-1,1)
    for c in range(opt.n_classes):
        pts = real_2d[labels==c]
        col = rand_colour()
        for x,y in pts:
            plt.text(x, y, str(c), color=col, alpha=0.5,
                     fontdict={'weight':'bold','size':9})
    plt.title(f"{opt.dataset_name} classifier result (epoch={epoch})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"classifier_result",
                             f"classifier_{epoch}.jpeg"))
    plt.close()

# -------------------- PCM helper --------------------------
def pcm_labels(feat_real, feat_gen):
    """
    Runs PCM separately on real & generated features (dim×N),
    then matches centres by nearest-neighbour. Returns
    long-dtype labels for the real batch.
    """
    c, m, eps, itmax = (opt.n_classes, opt.pcm_m,
                        opt.pcm_e, opt.pcm_max_iter)

    # === generated ===
    vg,_,ug,_,_,_ = pcm(feat_gen.T, c, m, eps, itmax)
    # === real ===
    vr,_,ur,_,_,_ = pcm(feat_real.T, c, m, eps, itmax)

    # nearest centre mapping
    cost = torch.cdist(torch.tensor(vr), torch.tensor(vg))
    mapping = cost.argmin(1).cpu().numpy()        # real-cluster-i  → gen-cluster-j*

    r_hard = ur.argmax(0)                         # (Nb,)
    r_mapped = np.vectorize(lambda lab: mapping[lab])(r_hard)
    return torch.tensor(r_mapped, device=device, dtype=torch.long)

# -------------------- Training ----------------------------
each  = 20                    # samples per class for snapshots
alpha = 0.0
d_log, g_log, i_log = [], [], []

for epoch in range(opt.n_epochs):
    for (real,) in loader:
        real = real.to(device)
        B    = real.size(0)

        # ==================================================
        #        (1)  Adversarial step  (D & G)
        # ==================================================
        z_adv   = torch.randn(B, opt.latent_dim, device=device)
        y_adv   = torch.randint(0, opt.n_classes, (B,), device=device)
        oh_adv  = to_onehot(y_adv, opt.n_classes)

        # ---- Generator ----------------------------------
        opt_G.zero_grad()
        gen_adv              = G(z_adv, oh_adv)               # fresh forward
        adv_fake, _, _       = D(gen_adv)
        g_loss               = adv_loss(adv_fake,
                                        torch.ones_like(adv_fake))
        g_loss.backward()
        opt_G.step()

        # ---- Discriminator ------------------------------
        opt_D.zero_grad()
        adv_real, _, _       = D(real)
        adv_fake, _, _       = D(gen_adv.detach())
        d_loss = 0.5*(adv_loss(adv_real, torch.ones_like(adv_real)) +
                      adv_loss(adv_fake, torch.zeros_like(adv_fake)))
        d_loss.backward()
        opt_D.step()

        # ==================================================
        #        (2)  Info / PCM step  (G + D together)
        # ==================================================
        opt_I.zero_grad()

        # --- *new* forward pass through G ----------------
        z_info   = torch.randn(each*opt.n_classes,
                               opt.latent_dim, device=device)
        tgt_info = torch.arange(opt.n_classes, device=device)\
                        .repeat_interleave(each)
        oh_info  = to_onehot(tgt_info, opt.n_classes)
        gen_info = G(z_info, oh_info)

        _, cls_g,  feat_g = D(gen_info)
        _, cls_r,  feat_r = D(real)

        g_cls_loss = cls_loss(cls_g, tgt_info)

        # --------- PCM on **detached** features ----------
        r_labels = pcm_labels(F.normalize(feat_r,1).detach().cpu().numpy(),
                              F.normalize(feat_g,1).detach().cpu().numpy())
        r_cls_loss = cls_loss(cls_r, r_labels)

        info_loss = g_cls_loss + alpha * r_cls_loss
        info_loss.backward()
        opt_I.step()

    # ─ logging & α schedule ─
    alpha = 0.001 if epoch <= 0.7*opt.n_epochs else 0.1
    d_log.append(d_loss.item()); g_log.append(g_loss.item())
    i_log.append(info_loss.item())
    print(f"[{epoch:4d}/{opt.n_epochs}] "
          f"D:{d_loss.item():.4f} G:{g_loss.item():.4f} "
          f"Info:{info_loss.item():.4f}")

    # -------- snapshot / plots --------------------------
    if epoch % opt.sample_interval == 0:
        torch.save(G.state_dict(),
                   os.path.join(outdir,"generator_model",f"gen_{epoch}.pth"))
        torch.save(D.state_dict(),
                   os.path.join(outdir,"discriminator_model",f"disc_{epoch}.pth"))
        with torch.no_grad():
            vis = G(z_info, oh_info).cpu()           # reuse newest graph-free
        plot_pca(vis, epoch)
        plot_classifier(epoch)


# -------------------- Curves ------------------------------
plt.figure(figsize=(10,5))
plt.plot(d_log, label="D loss", alpha=.7)
plt.plot(g_log, label="G loss", alpha=.7)
plt.plot(i_log, label="GC loss", alpha=.7)
plt.title(f"Learning curves – {opt.dataset_name} (PCM)")
plt.xlabel("iterations"); plt.ylabel("loss"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir,f"{opt.dataset_name}_learning_curves.png"))
plt.close()

print("Finished – all results stored in", outdir)
