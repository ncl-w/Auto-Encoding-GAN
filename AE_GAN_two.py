import argparse
import os
import random

import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


# ----------------------------------------------------------
# 1. DEVICE SETUP (works for mac MPS, CUDA, or CPU)
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
parser.add_argument("--n_epochs", type=int, default=500,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2,
                    help="latent code")
parser.add_argument("--n_classes", type=int, default=15,
                    help="number of classes (modes) for dataset")
parser.add_argument("--img_size", type=int, default=32,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10,
                    help="interval between image sampling")
opt = parser.parse_args()

print(opt)

# ----------------------------------------------------------
# 3. OUTPUT FOLDERS
# ----------------------------------------------------------
# We'll make a top folder like "<dataset_name>_images"
file_name = f"{opt.dataset_name}_images"
os.makedirs(file_name, exist_ok=True)

root = os.getcwd()
generator_path = os.path.join(root, file_name, "generator_model")
os.makedirs(generator_path, exist_ok=True)

discriminator_path = os.path.join(root, file_name, "discriminator_model")
os.makedirs(discriminator_path, exist_ok=True)

# For visualization:
test_path = os.path.join(root, file_name, "test_result")
os.makedirs(test_path, exist_ok=True)

classifier_path = os.path.join(root, file_name, "classifier_result")
os.makedirs(classifier_path, exist_ok=True)

# ----------------------------------------------------------
# 4. INITIALIZATION FUNCTIONS
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
                                       nn.Softmax(dim=1)) # dim =1 to surpress the warning
        self.latent_layer = nn.Sequential(nn.Linear(ds_size, opt.code_dim))

    def forward(self, img):
        feature_out = self.model(img)
        validity = self.adv_layer(feature_out)
        label = self.aux_layer(feature_out)
        return validity, label, feature_out

# ----------------------------------------------------------
# 6. VISUALIZATION HELPERS
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

def feature_space_kmeans_cluster(gen_feature_dataset, real_feature_dataset, k, each_gen_num):
    gen_np = gen_feature_dataset.cpu().detach().numpy()
    real_np = real_feature_dataset.cpu().detach().numpy()

    gen_cluster_model = KMeans(n_clusters=k, n_init='auto') # n_init='auto' to surpress the warning
    gen_kmeans_pred = gen_cluster_model.fit_predict(gen_np)
    gen_centers = gen_cluster_model.cluster_centers_

    real_cluster_model = KMeans(n_clusters=k, n_init='auto') # n_init='auto' to surpress the warning
    real_kmeans_pred = real_cluster_model.fit_predict(real_np)
    real_centers = real_cluster_model.cluster_centers_

    # Calculate pairwise cos similarity
    centers_matrix = F.cosine_similarity(
        torch.tensor(real_centers),
        torch.tensor(gen_centers),
        dim=1
    )

    adjust_real_kmeans_pred = np.zeros(real_kmeans_pred.shape[0], dtype=np.int32) - 1
    for i in range(centers_matrix.shape[0]):
        index = torch.argmax(centers_matrix[i])
        adjust_real_kmeans_pred[np.where(real_kmeans_pred == i)] = index

    real_kmeans_pred = adjust_real_kmeans_pred

    clusters = np.unique(gen_kmeans_pred)
    gen_y_pred = np.zeros(gen_kmeans_pred.shape[0], dtype=np.int32)
    real_y_pred = np.zeros(real_kmeans_pred.shape[0], dtype=np.int32)

    for i in range(clusters.shape[0]):
        y_ = gen_kmeans_pred[i * each_gen_num : (i+1) * each_gen_num]
        cluster_label = np.argmax(np.bincount(y_))
        gen_y_pred[i * each_gen_num : (i+1) * each_gen_num] = i
        real_y_pred[np.where(real_kmeans_pred == cluster_label)] = i

    return gen_centers, real_y_pred

# ----------------------------------------------------------
# 7. SETUP MODELS, DATA, OPTIMIZERS
# ----------------------------------------------------------
adversarial_loss = nn.BCELoss().to(device)
categorical_loss = nn.CrossEntropyLoss().to(device)
continuous_loss = nn.MSELoss().to(device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# LOAD the specified dataset
dataset_file = os.path.join(root, "dataset", f"{opt.dataset_name}.txt")
print("Loading dataset file:", dataset_file)
X = np.loadtxt(dataset_file)[:, 0:2]
X = Normalization(X, -0.8, 0.8)
dataset_torch = torch.tensor(X, dtype=torch.float, device=device)
drop_last_batch = (opt.dataset_name == "S1") # S1 - drop last batch to avoid size mismatch in Kmeans

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
# 8. LOGGING LOSSES
# ----------------------------------------------------------
d_losses, g_losses, gc_losses = [], [], []

# ----------------------------------------------------------
# 9. TRAINING LOOP
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

        # -----------------
        #  Train Generator
        # -----------------
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

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        d_loss = 0.5 * (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Train Info (Generator + Real)
        # ------------------
        optimizer_info.zero_grad()

        gen_imgs = generator(z, mutil_one_hot_code)
        _, gen_pred_label, gen_feature = discriminator(gen_imgs)
        _, real_pred_label, real_feature = discriminator(real_imgs)

        gen_cluster_loss = categorical_loss(gen_pred_label, mutil_gen_target)

        gen_feature = F.normalize(gen_feature, dim=1)
        real_feature = F.normalize(real_feature, dim=1)

        centers, real_cluster_target = feature_space_kmeans_cluster(
            gen_feature, real_feature, opt.n_classes, each_gen_num
        )

        real_cluster_target = torch.tensor(real_cluster_target, dtype=torch.long, device=device)
        real_cluster_loss = categorical_loss(real_pred_label, real_cluster_target)

        gc_loss = gen_cluster_loss + alpha * real_cluster_loss
        gc_loss.backward()
        optimizer_info.step()

        # Log & Print
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        gc_losses.append(gc_loss.item())

        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
              f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [gc loss: {gc_loss.item():.6f}]")

    # alpha scheduling
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
# 10. AFTER TRAINING: PLOT LEARNING CURVES
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label="D loss", alpha=0.7)
plt.plot(g_losses, label="G loss", alpha=0.7)
plt.plot(gc_losses, label="GC loss", alpha=0.7)
plt.title(f"Learning Curves - {opt.dataset_name}")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plot_file = os.path.join(file_name, f"{opt.dataset_name}_learning_curves.png")
plt.savefig(plot_file)
plt.close()

print(f"Training complete. Plots saved to: {plot_file}")
