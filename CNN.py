import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, hidden_channel=32, output_channel=32, output_size=64):
        super().__init__()

        self.conv_layer1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_channel,
            kernel_size=5,
            stride=2,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=1,
        )
        self.conv_layer2 = nn.Conv2d(
            in_channels=hidden_channel,
            out_channels=output_channel,
            kernel_size=5,
            stride=2
        )
        self.adaptive_pool = nn.AdaptiveMaxPool2d(
            output_size=(output_size, output_size)
        )
        self.cnn = nn.Sequential(
            self.conv_layer1,
            nn.ReLU(),
            self.max_pool,
            self.conv_layer2,
            nn.ReLU(),
            self.adaptive_pool,
        )

    def forward(self, img):

        return self.cnn(img)

def visualize_feature_map(feat, num_channels=8):
    # (C, H, W)
    C = feat.shape[0]
    # 选择要显示的通道索引
    channels = np.arange(min(num_channels, C))
    fig, axes = plt.subplots(1, len(channels), figsize=(15, 5))
    if len(channels) == 1:
        axes = [axes]
    for i, idx in enumerate(channels):
        # 取出第 idx 个通道，并归一化到 [0,1] 便于显示
        channel = feat[idx].detach().numpy()
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        axes[i].imshow(channel, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {idx}')
    plt.show()

if __name__ == "__main__":

    mean = 251.7802
    std = 28.4787

    image = Image.open("mol_img/test/0/0/0/000037687605.png")
    arr = np.array(image, dtype=np.float32)
    arr -= mean
    arr /= std
    arr = - arr
    img_tensor = torch.tensor(arr, dtype=torch.float32)
    h, w = img_tensor.shape
    img_tensor = img_tensor.reshape([-1, h, w])
    #visualize_feature_map(img_tensor)
    print(img_tensor.shape)
    cnn = CNN()
    cnn.eval()
    output_img = cnn(img_tensor.unsqueeze(1))
    print(output_img.shape)
    #visualize_feature_map(output_img[0], 8)




