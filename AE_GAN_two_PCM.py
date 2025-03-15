#!/usr/bin/env python3
import argparse
import os
import random

import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
# Instead of KMeans, we will use PCM from cmeans.py
# from sklearn.cluster import KMeans
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# Import your PCM function
from cmeans import pcm  # <-- must be in same directory or installed as a package

# ----------------------------------------------------------
# 1. DEVICE SETUP
# ----------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda:0" if torch.cuda.is_available()
                      else "cpu")

# ----------------------------------------------------------
# 2. PARSE ARGUMENTS
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="R15",
                    help="Name of the dataset: 'R15', 'S1', 'Aggregation', etc.")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=15, help="number of classes (modes)")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")

# Additional hyperparams for PCM
parser.add_argument("--pcm_m", type=float, default=2.0, help="PCM fuzzifier (m)")
parser.add_argument("--pcm_e", type=float, default=1e-4, help="PCM convergence threshold")
parser.add_argument("--pcm_max_iter", type=int, default=50, help="Max PCM iterations")

opt = parser.parse_args()
print(opt)

# ----------------------------------------------------------
# 3. OUTPUT FOLDERS
# ----------------------------------------------------------
file_name = f"{opt.dataset_name}_images_PCM"
os.makedirs(file_name, exist_ok=True)

root = os.getcwd()
generator_path = os.path.join(root, file_name, "generator_model")
os.makedirs(generator_path, exist_ok=True)

discriminator_path = os.path.join(root, file_name, "discriminator_model")
os.makedirs(discriminator_path, exist_ok=True)

test_path = os.path.join(root, file_name, "test_result")
os.makedirs(test_path, exist_ok=True)

classifier_path = os.path.join(root, file_name, "classifier_result")
os.makedirs(classifier_path, exist_ok=True)

# ----------------------------------------------------------
# 4. INIT FUNCTIONS
# ----------------------------------------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def to_categorical(y, num_columns):
    y_cat = torch.zeros((y.shape[0], num_columns), device=device)
    y_cat[range(y.shape[0]), y] = 1.0
    return y_cat

def randomcolor():
    colorArr = list("123456789ABCDEF")
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def Normalization(dataset, min_value, max_value):
    dim_min = np.min(dataset)
    dim_max = np.max(dataset)
    if abs(dim_max - dim_min) <= 1e-6:
        return dataset
    else:
        return ((dataset - dim_min) / (dim_max - dim_min)) * (max_value - min_value) + min_value

# ----------------------------------------------------------
# 5. MODEL CLASSES
# ----------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, 0.8),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.BatchNorm1d(2, 0.8),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        gen_input = torch.cat((z, labels), -1)
        return self.model(gen_input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, 0.8),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, 0.8),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        ds_size = 512
        self.adv_layer = nn.Sequential(nn.Linear(ds_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(ds_size, opt.n_classes),
                                       nn.Softmax(dim=1))
        self.latent_layer = nn.Sequential(nn.Linear(ds_size, opt.code_dim))

    def forward(self, img):
        feature_out = self.model(img)
        validity = self.adv_layer(feature_out)
        label = self.aux_layer(feature_out)
        return validity, label, feature_out

# ----------------------------------------------------------
# 6. VISUALIZATION
# ----------------------------------------------------------
def ShowImage(dataset, total_gen_data, each_gen_num, epoch):
    dataset_np = dataset.cpu().detach().numpy()
    total_np = total_gen_data.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)

    # Plot real data in blue
    plt.scatter(dataset_np[:, 0], dataset_np[:, 1], color='b', marker='.')

    # Plot generated data in text form, color-coded per class
    for g in range(opt.n_classes):
        gen_data = total_np[g*each_gen_num : (g+1)*each_gen_num]
        color = randomcolor()
        for l in range(gen_data.shape[0]):
            plt.text(gen_data[l, 0], gen_data[l, 1], str(g),
                     color=color, fontdict={'weight': 'bold', 'size': 9})

    plt.title(f"{opt.dataset_name} - epoch {epoch}")
    out_file = os.path.join(test_path, f"gan_{epoch}.jpg")
    plt.savefig(out_file)
    plt.close()

def classifier_result(dataset, y, cluster_num, epoch):
    dataset_np = dataset.cpu().detach().numpy()
    y_np = y.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)

    # For each cluster, plot text
    for j in range(cluster_num+1):
        color = randomcolor()
        for i in range(dataset_np.shape[0]):
            if int(y_np[i]) == j:
                plt.text(dataset_np[i, 0], dataset_np[i, 1], str(int(y_np[i])),
                         color=color, fontdict={'weight': 'bold', 'size': 9}, alpha=0.5)

    plt.title(f"{opt.dataset_name} classifier result (epoch={epoch})")
    out_file = os.path.join(classifier_path, f"image_{epoch}.jpeg")
    plt.savefig(out_file)
    plt.close()

# ----------------------------------------------------------
# 7. PCM-BASED CLUSTERING
# ----------------------------------------------------------
def feature_space_pcm_cluster(gen_feature_dataset, real_feature_dataset, k, each_gen_num):
    """
    Uses PCM to cluster the generated features and real features, then does
    a simple bipartite matching of cluster centers. Includes debug prints to
    confirm shapes and detect out-of-range indices.
    """
    from cmeans import pcm  # Must have cmeans.py in the same folder

    # Convert PyTorch tensors => NumPy in shape (dim, N)
    gen_np = gen_feature_dataset.cpu().detach().numpy().T
    real_np = real_feature_dataset.cpu().detach().numpy().T

    c = k
    m = opt.pcm_m         # e.g., 2.0
    e = opt.pcm_e         # e.g., 1e-4
    maxit = opt.pcm_max_iter  # e.g., 50

    # 1) PCM on generated data
    gen_v, _, gen_u, _, _, _ = pcm(gen_np, c, m, e, maxit, metric="euclidean")
    # 2) PCM on real data
    real_v, _, real_u, _, _, _ = pcm(real_np, c, m, e, maxit, metric="euclidean")

    # Check shapes (important!)
    # print(f"gen_v.shape: {gen_v.shape}")    # Expect (k, dim) => e.g. (5, 512)
    # print(f"real_v.shape: {real_v.shape}")  # Also (k, dim)

    # Convert these cluster centers to Torch but DO NOT transpose
    real_v_t = torch.tensor(real_v)  # shape => (k, dim)
    gen_v_t  = torch.tensor(gen_v)   # shape => (k, dim)

    # We'll do a cost matrix of shape (k, k)
    cost_matrix = torch.cdist(real_v_t, gen_v_t, p=2.0)
    # print("cost_matrix.shape:", cost_matrix.shape)

    # Hard label each real sample by argmax of real_u
    real_hard_labels = np.argmax(real_u, axis=0)  # shape (Nreal,)

    # We'll unify real cluster i with whichever gen cluster j is minimal cost
    # and re-assign real points from cluster i => j
    adjust_real_labels = np.zeros_like(real_hard_labels) - 1

    for i in range(k):
        row_i = cost_matrix[i]  # shape (k,)
        j_star = torch.argmin(row_i).item()
        # if j_star >= k:
        #     print(f"ERROR: j_star {j_star} out of range for i={i}")
        # Now all real points that had label i => j_star
        idx = np.where(real_hard_labels == i)[0]
        adjust_real_labels[idx] = j_star

    # For the generated side, we do a simple "argmax" across gen_u
    gen_labels = np.argmax(gen_u, axis=0)

    # Return final cluster labels
    # gen_v = cluster centers for generated data
    # real_y_pred = the new real cluster label
    # The user code can do `torch.tensor(real_y_pred, dtype=torch.long, ...)`
    return gen_v, adjust_real_labels


# ----------------------------------------------------------
# 8. SETUP MODELS, DATA, OPTIMIZERS
# ----------------------------------------------------------
adversarial_loss = nn.BCELoss().to(device)
categorical_loss = nn.CrossEntropyLoss().to(device)
continuous_loss = nn.MSELoss().to(device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

dataset_file = os.path.join(root, "dataset", f"{opt.dataset_name}.txt")
print("Loading dataset file:", dataset_file)
X = np.loadtxt(dataset_file)[:, 0:2]
X = Normalization(X, -0.8, 0.8)
dataset_torch = torch.tensor(X, dtype=torch.float, device=device)

drop_last_batch = (opt.dataset_name == "S1")
dataloader = DataLoader(
    dataset=dataset_torch,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=drop_last_batch
)
print(f"Using drop_last={drop_last_batch} for dataset {opt.dataset_name}")

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()),
    lr=opt.lr, betas=(opt.b1, opt.b2)
)

# ----------------------------------------------------------
# 9. LOGGING LOSSES
# ----------------------------------------------------------
d_losses, g_losses, gc_losses = [], [], []

# ----------------------------------------------------------
# 10. TRAINING LOOP
# ----------------------------------------------------------
alpha = 0.0
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        batch_size = imgs.shape[0]
        each_gen_num = 10

        valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
        real = torch.ones(each_gen_num * opt.n_classes, 1, device=device, requires_grad=False)
        fake = torch.zeros(each_gen_num * opt.n_classes, 1, device=device, requires_grad=False)

        real_imgs = imgs

        # ----------------- Train Generator -----------------
        optimizer_G.zero_grad()

        z = torch.randn(each_gen_num * opt.n_classes, opt.latent_dim, device=device)
        # Make generator labels
        mutil_gen_target = torch.empty((0, 1), dtype=torch.long, device=device)
        for j in range(opt.n_classes):
            gen_target = torch.zeros((each_gen_num, 1), dtype=torch.long, device=device) + j
            mutil_gen_target = torch.cat((mutil_gen_target, gen_target), dim=0)
        mutil_gen_target = mutil_gen_target.view(-1)
        mutil_one_hot_code = to_categorical(mutil_gen_target, opt.n_classes)

        gen_imgs = generator(z, mutil_one_hot_code)
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, real)

        g_loss.backward()
        optimizer_G.step()

        # ----------------- Train Discriminator -----------------
        optimizer_D.zero_grad()

        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        d_loss = 0.5 * (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # ------------------ Info / cluster alignment ------------------
        optimizer_info.zero_grad()

        gen_imgs = generator(z, mutil_one_hot_code)
        _, gen_pred_label, gen_feature = discriminator(gen_imgs)
        _, real_pred_label, real_feature = discriminator(real_imgs)

        gen_cluster_loss = categorical_loss(gen_pred_label, mutil_gen_target)

        gen_feature = F.normalize(gen_feature, dim=1)
        real_feature = F.normalize(real_feature, dim=1)

        # *** Use PCM-based clustering ***
        centers, real_cluster_target = feature_space_pcm_cluster(
            gen_feature, real_feature, opt.n_classes, each_gen_num
        )

        real_cluster_target = torch.tensor(real_cluster_target, dtype=torch.long, device=device)
        # print("real_cluster_target min:", real_cluster_target.min(), "max:", real_cluster_target.max())
        # print("unique:", np.unique(real_cluster_target))
        real_cluster_loss = categorical_loss(real_pred_label, real_cluster_target)
        


        gc_loss = gen_cluster_loss + alpha * real_cluster_loss
        gc_loss.backward()
        optimizer_info.step()

        # Logging
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        gc_losses.append(gc_loss.item())

        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
              f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [gc loss: {gc_loss.item():.6f}]")

    if epoch <= int(opt.n_epochs * 0.7):
        alpha = 0.001
    else:
        alpha = 0.1

    # Save images / classifier result
    if epoch % opt.sample_interval == 0:
        each_gen_num = 20
        z = torch.randn(each_gen_num * opt.n_classes, opt.latent_dim, device=device)
        gen_targets = torch.empty((0, 1), dtype=torch.long, device=device)
        for j in range(opt.n_classes):
            gtarget = torch.zeros((each_gen_num, 1), dtype=torch.long, device=device) + j
            gen_targets = torch.cat((gen_targets, gtarget), dim=0)
        gen_targets = gen_targets.view(-1)
        one_hot_code = to_categorical(gen_targets, opt.n_classes)

        gen_imgs2 = generator(z, one_hot_code)
        ShowImage(dataset_torch, gen_imgs2, each_gen_num, epoch)

        c_out = discriminator(dataset_torch)[1]
        y_pred = torch.max(c_out, 1)[1]
        print("Classifier Predicted Labels", torch.unique(y_pred))

        classifier_result(dataset_torch, y_pred, opt.n_classes, epoch)

        # Save model
        discriminator_model_dir = os.path.join(discriminator_path, f"discriminator_{epoch}.pth")
        torch.save({'model': discriminator.state_dict()}, discriminator_model_dir)

        generator_model_dir = os.path.join(generator_path, f"generator_{epoch}.pth")
        torch.save({'model': generator.state_dict()}, generator_model_dir)

# ----------------------------------------------------------
# 11. AFTER TRAINING: PLOT LEARNING CURVES
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label="D loss", alpha=0.7)
plt.plot(g_losses, label="G loss", alpha=0.7)
plt.plot(gc_losses, label="GC loss", alpha=0.7)
plt.title(f"Learning Curves - {opt.dataset_name} (PCM-based Clustering)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plot_file = os.path.join(file_name, f"{opt.dataset_name}_learning_curves.png")
plt.savefig(plot_file)
plt.close()

print(f"Training complete. Plots saved to: {plot_file}")
