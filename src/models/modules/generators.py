import torch
import torch.nn as nn
import numpy as np


# class Generator(nn.Module):
#     def __init__(
#             self,
#             n_classes: int,
#             latent_dim: int,
#             channels: int,
#             img_size: int,
#     ):
#         super().__init__()
#         self.img_shape = (channels, img_size, img_size)
#         self.label_emb = nn.Embedding(n_classes, n_classes)
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *block(latent_dim + n_classes, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(self.img_shape))),
#             nn.Tanh()
#         )
#
#     def forward(self, noise, labels):
#         # Concatenate label embedding and image to produce input
#         gen_input = torch.cat((self.label_emb(labels), noise), -1)
#         img = self.model(gen_input)
#         img = img.view(img.size(0), *self.img_shape)
#         return img


class Generator(nn.Module):
    def __init__(self, n_classes: int, latent_dim: int, channels: int, img_size: int):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size

        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.bias = False

        self.img_shape = (channels, img_size, img_size)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.init_size = img_size // 32

        self.linear = nn.Linear(latent_dim + n_classes, self.latent_dim * self.init_size ** 2)

        self.model = nn.Sequential(
            # input size: (100, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.25),

            # input size: (512, 2, 2)
            nn.ConvTranspose2d(512, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.25),

            # input size: (256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.25),

            # input size: (128, 8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.25),

            # input size: (64, 16, 16)
            nn.ConvTranspose2d(64, 1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               bias=self.bias),
            nn.Tanh(),
            # output size: (1, 32, 32)
        )

    def forward(self, noise, labels):
        # noise: (32, 100), label: (32)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        # print(gen_input.shape)  # (32, 110)
        out = self.linear(gen_input)
        # print(out.shape)  # (32, 100)
        out = out.view(out.shape[0], self.latent_dim, self.init_size, self.init_size)
        # print(out.shape)  # (32, 100, 1, 1)
        img = self.model(out)
        # print(img.shape)  # (32, 1, 32, 32)
        return img
