import torch
import torch.nn as nn

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dconv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Unet Integration
class DownBlockA(nn.Module):
    def __init__(self):
        super(DownBlockA, self).__init__()
        # Define the convolutional layer and pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3,
                               kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define the activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutional layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        return x


class DownBlockB(nn.Module):
    def __init__(self):
        super(DownBlockB, self).__init__()
        # Define the convolutional layer and pool
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define the activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutional layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        return x

# # Example usage:
# # Create an instance of the DownBlock
# cnn_layer = DownBlock()

# # Generate random input tensor with shape (batch_size, channels, height, width)
# input_tensor = torch.randn(1, 3, 512, 512)

# # Forward pass through the CNN layer
# output_tensor = cnn_layer(input_tensor)

# # Print the output shape
# print("Output shape:", output_tensor.shape)  # Should be (1, 3, 256, 256)


class UNetExperimental(nn.Module):
    """ Full assembly of the parts to form the complete network """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetExperimental, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.downBlockA = DownBlockA()
        self.downBlockB = DownBlockB()

        # Unet Light Module A
        self.initConvA = (DConv(n_channels, 64))
        self.down1A = (Down(64, 128))
        self.down2A = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down3A = (Down(256, 512))
        self.down4A = (Down(512, 1024 // factor))
        self.up1A = (Up(1024, 512 // factor, bilinear))
        self.up2A = (Up(512, 256 // factor, bilinear))
        self.up3A = (Up(256, 128 // factor, bilinear))
        self.up4A = (Up(128, 64, bilinear))
        self.outcA = (FinalConv(64, n_classes))

        # Unet Light Module B
        self.initConvB = (DConv(n_channels, 64))
        self.down1B = (Down(64, 128))
        self.down2B = (Down(128, 256))
        factorB = 2 if bilinear else 1
        self.down3B = (Down(256, 512))
        self.down4B = (Down(512, 1024 // factor))
        self.up1B = (Up(1024, 512 // factor, bilinear))
        self.up2B = (Up(512, 256 // factor, bilinear))
        self.up3B = (Up(256, 128 // factor, bilinear))
        self.up4B = (Up(128, 64, bilinear))
        self.outcB = (FinalConv(64, n_classes))
        self.upoutb = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=2, stride=2
        )

        # Unet Light Module C
        self.initConvC = (DConv(n_channels, 64))
        self.down1C = (Down(64, 128))
        self.down2C = (Down(128, 256))
        factorC = 2 if bilinear else 1
        self.down3C = (Down(256, 512))
        self.down4C = (Down(512, 1024 // factor))
        self.up1C = (Up(1024, 512 // factor, bilinear))
        self.up2C = (Up(512, 256 // factor, bilinear))
        self.up3C = (Up(256, 128 // factor, bilinear))
        self.up4C = (Up(128, 64, bilinear))
        self.outcC = (FinalConv(64, n_classes))
        self.upoutc = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=4
        )

        # Final Mask
        self.conv = nn.Conv2d(in_channels=3, out_channels=1,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Module A
        a1 = self.initConvA(x)
        a2 = self.down1A(a1)
        a3 = self.down2A(a2)
        a4 = self.down3A(a3)
        a5 = self.down4A(a4)
        a = self.up1A(a5, a4)
        a = self.up2A(a4, a3)
        a = self.up3A(a, a2)
        a = self.up4A(a, a1)
        a = self.outcA(a)

        # Module B
        x2 = self.downBlockA(x)
        b1 = self.initConvB(x2)
        b2 = self.down1B(b1)
        b3 = self.down2B(b2)
        b4 = self.down3B(b3)
        b5 = self.down4B(b4)
        b = self.up1B(b5, b4)
        b = self.up2B(b4, b3)
        b = self.up3B(b, b2)
        b = self.up4B(b, b1)
        b = self.outcB(b)
        # Add the output of module A to the output of module B
        b = self.upoutb(b)

        # Module C
        x3 = self.downBlockB(x)
        c1 = self.initConvC(x3)
        c2 = self.down1C(c1)
        c3 = self.down2C(c2)
        c4 = self.down3C(c3)
        c5 = self.down4C(c4)
        c = self.up1C(c5, c4)
        c = self.up2C(c4, c3)
        c = self.up3C(c, c2)
        c = self.up4C(c, c1)
        c = self.outcC(c)
        # Add the output of module B to the output of module C
        c = self.upoutc(c)
        # Concatenate the outputs of module A, B, and C along the channel dimension
        mask = torch.cat([a, b, c], dim=1)
        return self.conv(mask)  # Return the final mask

    def use_checkpointing(self):
        self.initConv = torch.utils.checkpoint(self.initConv)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    # # Generate random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(2, 3, 512, 512)

    # # Forward pass through the CNN layer
    output_tensor = DownBlockB()(input_tensor)
    out = UNetExperimental(n_channels=3, n_classes=1)(input_tensor)

    # # Print the output shape
    print("Output shape:", out.shape)  # Should be (1, 3, 256, 256)
