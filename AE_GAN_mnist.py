import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from sklearn.cluster import KMeans

import torch.nn as nn
import torch.nn.functional as F
import torch

# ----------------------------
#  CHANGED: Unify device usage
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

root = os.getcwd()
os.makedirs("mnist_images/static/", exist_ok=True)
img_path = root + "/mnist_images/" + "images"
isExists = os.path.exists(img_path)
if not isExists:
    os.mkdir(img_path)

discriminator_path = root + "/mnist_images/" + "mnist_discriminator_model"
isExists = os.path.exists(discriminator_path)
if not isExists:
    os.mkdir(discriminator_path)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# -------------------------------------
#  REMOVED: old "cuda = True if ... else"
#  Instead rely on device = ...
# -------------------------------------

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns), device=device)  # CHANGED: create on 'device'
    y_cat[range(y.shape[0]), y] = 1.0
    return y_cat


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, labels), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 512

        self.adv_layer = nn.Sequential(nn.Linear(ds_size, 1))
        self.aux_layer = nn.Sequential(nn.Linear(ds_size, opt.n_classes), nn.Softmax(dim=1)) # added dim=1 to surpress warning
        self.latent_layer = nn.Sequential(nn.Linear(ds_size, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label, out

def feature_space_kmeans_cluster(gen_feature_dataset, real_feature_dataset, k, each_gen_num):

    gen_feature_dataset = gen_feature_dataset.cpu().detach().numpy()
    real_feature_dataset = real_feature_dataset.cpu().detach().numpy()

    gen_cluster_model = KMeans(n_clusters=k, n_init='auto') # added n_init='auto' for compatibility with sklearn 1.0
    gen_kmeans_pred = gen_cluster_model.fit_predict(gen_feature_dataset)
    gen_centers = gen_cluster_model.cluster_centers_

    real_cluster_model = KMeans(n_clusters=k, n_init='auto') # added n_init='auto' for compatibility with sklearn 1.0
    real_kmeans_pred = real_cluster_model.fit_predict(real_feature_dataset)
    real_centers = real_cluster_model.cluster_centers_

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

    gen_y_pred = np.zeros(gen_kmeans_pred.shape[0], dtype=np.int32)
    real_y_pred = np.zeros(real_kmeans_pred.shape[0], dtype=np.int32)
    clusters = np.unique(gen_kmeans_pred)
    for i in range(clusters.shape[0]):
        y = gen_kmeans_pred[i * each_gen_num:(i + 1) * each_gen_num]
        cluster_label = np.argmax(np.bincount(y))
        gen_y_pred[i * each_gen_num:(i + 1) * each_gen_num] = i
        real_y_pred[np.where(real_kmeans_pred == cluster_label)] = i

    return gen_centers, real_y_pred


# Loss functions
adversarial_loss = torch.nn.MSELoss().to(device)  # CHANGED: move to device
categorical_loss = torch.nn.CrossEntropyLoss().to(device)  # CHANGED: move to device

# Initialize models
generator = Generator().to(device)     # CHANGED
discriminator = Discriminator().to(device)  # CHANGED

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)


def sample_image(n_row, batches_done, static_label, each_gen_num):
    """Saves a grid of generated digits"""
    z = torch.randn(each_gen_num * opt.n_classes, opt.latent_dim, device=device)  # CHANGED
    static_sample = generator(z, static_label)
    save_image(static_sample.data, "mnist_images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

alpha = 0.0
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        each_gen_num = 10  # number of samples per generator

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device, requires_grad=False)  # CHANGED
        real = torch.ones(each_gen_num * opt.n_classes, 1, device=device, requires_grad=False)  # CHANGED
        fake = torch.zeros(each_gen_num * opt.n_classes, 1, device=device, requires_grad=False) # CHANGED

        real_imgs = imgs.to(device)  # CHANGED: move images to device

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        z = torch.randn(each_gen_num * opt.n_classes, opt.latent_dim, device=device)  # CHANGED
        mutil_gen_target = torch.empty((0, 1), dtype=torch.long, device=device)  # CHANGED

        for j in range(opt.n_classes):
            gen_target = torch.zeros((each_gen_num, 1), dtype=torch.long, device=device)  # CHANGED
            gen_target = gen_target + j
            mutil_gen_target = torch.cat((mutil_gen_target, gen_target), dim=0)

        mutil_gen_target = mutil_gen_target.view(-1)
        mutil_one_hot_code = to_categorical(mutil_gen_target, opt.n_classes)  # create on device

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

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # gen + real Loss
        # ------------------
        optimizer_info.zero_grad()

        gen_imgs = generator(z, mutil_one_hot_code)
        _, gen_pred_label, gen_feature = discriminator(gen_imgs)
        _, real_pred_label, real_feature = discriminator(real_imgs)

        gen_cluster_loss = categorical_loss(gen_pred_label, mutil_gen_target)

        gen_feature = F.normalize(gen_feature, dim=1)  # CHANGED: specify dim
        real_feature = F.normalize(real_feature, dim=1)

        centers, real_cluster_target = feature_space_kmeans_cluster(
            gen_feature, real_feature, opt.n_classes, each_gen_num
        )

        real_cluster_target = torch.tensor(real_cluster_target, dtype=torch.long, device=device)  # CHANGED
        real_cluster_loss = categorical_loss(real_pred_label, real_cluster_target)

        gc_loss = gen_cluster_loss + alpha * real_cluster_loss
        gc_loss.backward()
        optimizer_info.step()

        print(
            f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
            f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [gc loss: {gc_loss.item():.6f}]"
        )

    if epoch <= int(opt.n_epochs * 0.7):
        alpha = 0.001
    else:
        alpha = 0.1

    batches_done = epoch
    if batches_done % opt.sample_interval == 0:
        sample_image(n_row=10, batches_done=batches_done, static_label=mutil_one_hot_code, each_gen_num=each_gen_num)

        discriminator_model_dir = os.path.join(discriminator_path, f"discriminator_{epoch}.pth")
        state = {'model': discriminator.state_dict()}
        torch.save(state, discriminator_model_dir)

# Extra final sampling
from torchvision import utils as vutils
z = torch.randn(256, opt.latent_dim, device=device)  # CHANGED
sampled_labels = np.random.randint(0, opt.n_classes, 256)
label_input = to_categorical(torch.LongTensor(sampled_labels).to(device), opt.n_classes)
gen_imgs = generator(z, label_input)
for i in range(gen_imgs.shape[0]):
    imgs_name = os.path.join(img_path, f"{i}.jpg")
    img = gen_imgs[i]
    vutils.save_image(img, imgs_name)
