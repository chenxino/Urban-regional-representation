import argparse
import os
import numpy as np
import math
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import pandas  as pd
from tkinter import _flatten
import random
import networkx as nx
import time

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--max_iters", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--struct", type=list, default=[14,12], help=" dimension of autoencoder network")
parser.add_argument("--dis_struct", type=list, default=[5,1], help=" dimension of discriminator network")
parser.add_argument("--fea_size", type=int, default=16, help="size of feature dimension")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")

opt = parser.parse_args(args=[])
print(opt)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        struct = [12, 8]
        struct = opt.struct
        self.encoder = nn.Sequential(nn.Linear(int(opt.fea_size), struct[0]),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(struct[0], struct[1]),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(struct[1], int(opt.latent_dim)),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.decoder = nn.Sequential(nn.Linear(int(opt.latent_dim), struct[1]),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(struct[1], struct[0]),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(struct[0], int(opt.fea_size)),
                                     nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        struct = opt.dis_struct
        self.model = nn.Sequential(
            nn.Linear(int(opt.latent_dim), struct[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(struct[0], struct[1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
def read_feature2(inputFileName):
    f = open(inputFileName, 'r')
    lines = f.readlines()
    f.close()
    features = []
    for line in lines[0:]:
        l = line.strip('\n\r').split(' ')
        features.append(l[0:])
    features = np.array(features, dtype=np.float32)
    return features


# PPMI矩阵
def four_ord_ppmi(matrix, node_size):
    def recip_mat(matrix):
        return np.reciprocal(np.sqrt(np.sum(matrix, axis=1)))

    dig_value = recip_mat(matrix).tolist()
    dig_value = list(_flatten(dig_value))
    trans_mat = np.mat(np.diag(dig_value))
    trans_A = trans_mat * matrix * trans_mat
    four_ord_mat = (
                               trans_A + trans_A * trans_A + trans_A * trans_A * trans_A + trans_A * trans_A * trans_A * trans_A) / 4
    four_ord_mat_value = recip_mat(four_ord_mat)
    dig_value2 = recip_mat(four_ord_mat).tolist()
    dig_value2 = list(_flatten(dig_value2))
    trans_mat2 = np.mat(np.diag(dig_value2))
    trans_B = trans_mat2 * four_ord_mat * trans_mat2
    ppmi_mat = np.log(trans_B) - np.log(1 / node_size)
    ppmi_mat[np.isnan(ppmi_mat)] = 0.0
    ppmi_mat[np.isinf(ppmi_mat)] = 0.0
    ppmi_mat[np.isneginf(ppmi_mat)] = 0.0
    ppmi_mat[ppmi_mat < 0] = 0.0
    return ppmi_mat


# PPMI，特征矩阵
def output_features_mat(trans_mat, feature_mat):
    node_num = feature_mat.shape[0]
    feature_num = feature_mat.shape[1]
    output_feature_mat = np.zeros_like(feature_mat)
    for i in range(node_num):
        trans_weight_sum = 0
        avg_features = np.zeros(shape=(feature_num))
        node_i = np.array(trans_mat[i])[0]
        node_i_index_array = np.nonzero(node_i)[0]
        node_i_index_len = len(node_i_index_array)
        for j in range(node_i_index_len):
            trans_index = node_i_index_array[j]
            trans_weight = node_i[trans_index]
            avg_features += trans_weight * feature_mat[trans_index]
            trans_weight_sum += trans_weight
        avg_features /= trans_weight_sum
        output_feature_mat[i] = avg_features
    return output_feature_mat

def tfidf_feature(df):
    # @param
    # R: total region numbers
    R = df.shape[0]
    points_vec = df
    feature_vec = (points_vec / points_vec.sum(axis=0)) * np.log10(R / np.count_nonzero(points_vec, axis=0))
    return feature_vec


def main(input_feature_mat, output_feature_mat):
    # 读文件
    # Read features
    print('reading features...')
    X = input_feature_mat
    N = X.shape[0]
    dims = opt.latent_dim
    X_target = output_feature_mat

    #     X=torch.from_numpy(X)
    #     X_target = torch.from_numpy(X_target)

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # ----------
    #  Training
    # ----------
    batch_size = opt.batch_size
    max_iters = opt.max_iters

    idx = 0
    print_every_k_iterations = 1
    start = time.time()

    loss_sg = 0
    loss_ae = 0
    loss_dis = 0
    loss_gene = 0

    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    criterion = torch.nn.MSELoss()

    # Optimizers
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=0.01)
    optimizer_G2 = torch.optim.Adam(generator.encoder.parameters(), lr=opt.lr)

    cuda = False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    for epoch in range(max_iters):
        idx += 1

        # train for autoencoder model
        start_idx = np.random.randint(0, N - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch_X = X[batch_idx]
        X_new = X_target[batch_idx]
        X_new = Variable(Tensor(X_new))

        batch_X = Variable(Tensor(batch_X))
        batch_Y, batch_XX = generator(batch_X)
        loss_ae_value = criterion(X_new, batch_XX)
        optimizer_G.zero_grad()

        loss_ae_value.backward(retain_graph=True)

        optimizer_G.step()

        loss_ae += loss_ae_value

        # train for discriminator
        z_real_dist = Variable(torch.Tensor(batch_size, dims).uniform_(0, 1))
        d_g_real = discriminator(z_real_dist)
        d_g_fake = discriminator(batch_Y.detach())

        alpha_g = Variable(Tensor(batch_size, dims).uniform_(0, 1))
        interpolates_g = alpha_g * d_g_real + ((1 - alpha_g) * d_g_fake)
        disc_g_interpolates = discriminator(interpolates_g)
        z = torch.ones(batch_size, 1)

        gradients_g = torch.autograd.grad(disc_g_interpolates, [interpolates_g], grad_outputs=z, create_graph=True)[0]
        slopes_g = gradients_g.pow(2).sum(1).sqrt()
        gradient_penalty_g = (slopes_g - 1).pow(2).sum()

        # Adversarial ground truths
        valid = Variable(Tensor(d_g_real.size()).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(d_g_fake.size()).fill_(0.0), requires_grad=False)

        dc_g_loss_real = adversarial_loss(d_g_real, valid)
        dc_g_loss_fake = adversarial_loss(d_g_fake, fake)
        dc_g_loss = 0.5 * (dc_g_loss_fake + dc_g_loss_real + gradient_penalty_g)
        loss_dis_value = dc_g_loss
        optimizer_D.zero_grad()
        loss_dis_value.backward(retain_graph=True)
        optimizer_D.step()

        loss_dis += loss_dis_value

        # train for generator
        generator_g_loss = 0.2 * adversarial_loss(d_g_fake, torch.ones_like(d_g_fake))
        loss_gene_value = generator_g_loss
        loss_gene_value.backward(retain_graph=True)
        optimizer_G2.step()
        loss_gene += loss_gene_value

        if idx % print_every_k_iterations == 0:
            end = time.time()
            print('iterations: %d' % idx + ', time elapsed: %.2f, ' % (end - start), end='')
            print(" autoencoder_loss:{}".format(loss_ae / idx))
            print("discriminator_loss:{}, generator_loss:{}".format(loss_dis / idx, loss_gene / idx))
            total_loss = loss_sg / idx + loss_ae / idx + loss_dis / idx + loss_gene / idx
            print('loss: %.2f, ' % total_loss, end='')

#             loss1.append(loss_ae / idx)
#             loss2.append(loss_dis / idx)
#             loss3.append(loss_gene / idx)
#             loss4.append(total_loss)

#     figsize = (10, 5)
#     fig = plt.figure(figsize=figsize)

#     plt.plot(loss1, color='red', label='loss_ae')
#     plt.plot(loss4, color='black', label='total_loss')

#     plt.legend(loc='best')
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.show()

#     figsize = (10, 5)
#     fig = plt.figure(figsize=figsize)

#     plt.plot(loss2, color='green', label='loss_dis')
#     plt.legend(loc='best')
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.show()

#     figsize = (10, 5)
#     fig = plt.figure(figsize=figsize)

#     plt.plot(loss3, color='blue', label='loss_gene')
#     plt.legend(loc='best')
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#     plt.show()

    print('optimization finished...')
    X = torch.from_numpy(X)
    X = torch.tensor(X, dtype=torch.float32)
    generator.eval()
    embedding_result, _ = generator(X)
    print(embedding_result)
    return embedding_result


if __name__ == '__main__':

    input_feature_mat = read_feature2('out.txt')
    print(input_feature_mat)
    input_feature_mat = tfidf_feature(input_feature_mat)
    print(input_feature_mat)
    print(input_feature_mat.shape)

    c4 = np.loadtxt("structure.txt")
    print(c4)
    print(c4.shape)
    input_mat = four_ord_ppmi(c4, input_feature_mat.shape[0])
    output_feature_mat = output_features_mat(input_mat, input_feature_mat)
    print(output_feature_mat)
    ee=main(input_feature_mat, output_feature_mat)
    np.savetxt('fea3.txt', ee)

