import torch
import torch.nn as nn

from .minibatch_discrimination import MinibatchDiscrimination


# class Discriminator(nn.Module):
#     def __init__(
#             self,
#             n_classes: int,
#             channels: int,
#             img_size: int,
#     ):
#         super().__init__()
#         self.img_shape = (channels, img_size, img_size)
#         self.label_embedding = nn.Embedding(n_classes, n_classes)
#
#         self.model = nn.Sequential(
#             nn.Linear(n_classes + int(np.prod(self.img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1),
#         )
#
#     def forward(self, img, labels):
#         # Concatenate label embedding and image to produce input
#         d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
#         validity = self.model(d_in)
#         return validity

class Discriminator(nn.Module):
    def __init__(self, n_classes: int, channels: int, img_size: int):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.img_size = img_size
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.bias = False
        self.negative_slope = 0.2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # input size: (1, 32, 32)
        self.img_shape = (channels, img_size, img_size)
        # Embedding for class labels
        self.label_embedding = nn.Embedding(n_classes, img_size * img_size)

        # first layer of conv
        self.conv1 = nn.Conv2d(self.channels + 1, 128, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(self.negative_slope, inplace=True)
        self.dropout = nn.Dropout(0.4)

        # second layer of conv
        self.conv2 = nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               bias=self.bias)
        self.bn2 = nn.BatchNorm2d(256)

        # third layer of conv
        self.conv3 = nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(512)

        # fourth layer of conv
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, bias=self.bias)
        self.bn4 = nn.BatchNorm2d(1024)

        self.minibatch_discrimination = MinibatchDiscrimination(in_features=4096, out_features=50, kernel_dims=5,
                                                                mean=True)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1),
            # nn.Linear(in_features=4146, out_features=1),  # minibatch discrimination output dim
            # nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, img, labels, epoch):
        # Add Gaussian noise to the input image, decay over epoch
        gaussian_noise = (torch.randn(img.shape, device=self.device) + 0.0) * (0.1/(epoch+1))
        img = img + gaussian_noise

        label_embedding = self.label_embedding(labels)
        # print(label_embedding.shape)  # (32, 32 * 32)
        label_embedding = label_embedding.view(-1, 1, self.img_size, self.img_size)
        # print(f"Label embedding shape: {label_embedding.shape}")  # (32, 1, 32, 32)
        # print(f"Image shape: {img.shape}")  # (32, 1, 32, 32)
        d_in = torch.cat((img, label_embedding), 1)
        # print(d_in.shape)  # (32, 2, 32, 32)
        validity = self.conv1(d_in)
        validity = self.bn1(validity)
        validity = self.dropout(validity)
        validity = self.relu(validity)
        # print(validity.shape)  # (32, 128, 16, 16)
        validity = self.conv2(validity)
        validity = self.bn2(validity)
        validity = self.dropout(validity)
        # print(validity.shape)  # (32, 256, 8, 8)
        validity = self.conv3(validity)
        validity = self.bn3(validity)
        validity = self.dropout(validity)
        # print(validity.shape)  # (32, 512, 4, 4)
        validity = self.conv4(validity)
        validity = self.bn4(validity)
        validity = self.relu(validity)
        # print(validity.shape)  # (32, 1024, 2, 2)
        validity = validity.view(validity.shape[0], -1)
        # print(validity.shape)  # (32, 4096)
        # validity = self.minibatch_discrimination(validity)
        # print(validity.shape)  # (32, 4146)
        validity = self.fc1(validity)
        # print(validity.shape)  # (32, 1)
        return validity
