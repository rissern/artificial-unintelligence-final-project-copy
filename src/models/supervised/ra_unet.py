import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad, sigmoid

'''
Residual Attention UNet builds off UNet.py 
'''
class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        """
        super().__init__()
        # if no mid_channels are specified, set mid_channels as out_channels
        if not mid_channels:
            mid_channels = out_channels
        # create a convolution from in_channels to mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 2, padding=1)
        # create a batch_norm2d of size mid_channels
        self.batch_norm_1 = nn.BatchNorm2d(mid_channels)
        # create a relu
        self.relu = relu
        # create a convolution from mid_channels to out_channels
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 2, padding=1)
        # create a batch_norm2d of size out_channels
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        # conv1
        x = self.conv1(x)
        # batch_norm1
        x = self.batch_norm_1(x)
        # relu
        x = self.relu(x)
        # conv2
        x = self.conv2(x)
        # batch_norm2
        x = self.batch_norm_2(x)
        # relu
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    """ Downscale using the maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # create a maxpool2d of kernel_size 2 and padding = 0
        self.maxpool2d = nn.MaxPool2d(2, padding=0)
        # create a doubleconvhelper
        self.doubleconv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x):
        # maxpool2d
        x = self.maxpool2d(x)
        # doubleconv
        x = self.doubleconv(x)
        return x

# given
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        # create up convolution using convtranspose2d from in_channels to in_channels//2
        self.conv = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride = 2)
        # use a doubleconvhelper from in_channels to out_channels
        self.doubleconv = DoubleConvHelper(in_channels, out_channels)
        

    def forward(self, x1, x2):
        # step 1 x1 is passed through the convtranspose2d
        x1 = self.conv(x1)
        # step 2 The difference between x1 and x2 is calculated to account for differences in padding
        # Check
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        merge = torch.cat([x2, x1], dim=1)
       
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # step 4 & 5
        # x2 represents the skip connection
        # Concatenate x1 and x2 together with torch.cat
        # merge = torch.cat((x1, x2))
        # step 6 Pass the concatenated tensor through a doubleconvhelper
        result = self.doubleconv(merge)
        # step 7 Return output 
        return result
    
class OutConv(nn.Module):
    """ OutConv is the replacement of the final layer to ensure
    that the dimensionality of the output matches the correct number of
    classes for the classification task.
    """
    def __init__(self, in_channels, out_channels):
        # added
        super().__init__()

        # create a convolution with in_channels = in_channels and out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 2)

    def forward(self, x):
        # evaluate x with the convolution
        # Unsure if this is correct
        x = self.conv(x)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.conv_x = nn.Conv2d(in_channels, out_channels, 2, stride=2)

        # G is gating signal
        self.conv_g = nn.Conv2d(in_channels, out_channels*2, 2)

        self.relu = relu

        self.psi = nn.Conv2d(in_channels, 1, 2)

        self.sigmoid = sigmoid

        self.upsample = nn.Upsample(size=(in_channels, out_channels))

        self.final_conv = nn.Conv2d(in_channels, out_channels, 2)
        # Might need another conv2D
        # Might add a batchnorm


    def forward(self, x, g):
        print("Att", x.shape, g.shape, self.in_c, self.out_c)

        res = self.conv_x(x)
        g = self.conv_g(g)

        res = torch.add(g, res)
        res = self.relu(res)
        res = self.psi(res)
        res = self.sigmoid(res)
        res = self.upsample(res)
        res = torch.mul(res, x)
        # res = self.final_conv(res)
        return res


class RA_UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.
        """
        super(RA_UNet, self).__init__()

        # save in_channels, out_channels, n_encoders, embedding_size to self
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_encoders = n_encoders
        self.embedding_size = embedding_size
        self.scale_factor = scale_factor

        # create a doubleconvhelper
        self.doubleconvhelper = DoubleConvHelper(in_channels, self.embedding_size)
        
        self.encoders = nn.ModuleList()
        self.activation_gates = nn.ModuleList()
        # for each encoder (there's n_encoders encoders)
        for _ in range(self.n_encoders):
            # append a new encoder with embedding_size as input and 2*embedding_size as output
            self.encoders.append(Encoder(self.embedding_size, self.embedding_size*2))
            self.activation_gates.append(AttentionGate(self.embedding_size, self.embedding_size*2))
            print("Encoder", self.embedding_size, self.embedding_size*2)
            # double the size of embedding_size
            self.embedding_size *= 2

        # store it in self.encoders as an nn.ModuleList
        
        self.decoders = nn.ModuleList()
        # for each decoder (there's n_encoders decoders)
        for i, _ in enumerate(range(self.n_encoders)):
        
            # if it's the last decoder
            if i == self.n_encoders-1:
                # create a decoder of embedding_size input and out_channels output
                self.decoders.append(Decoder(self.embedding_size, out_channels))

                print("Decoder", self.embedding_size, out_channels)
            else:
                # create a decoder of embeding_size input and embedding_size//2 output
                self.decoders.append(Decoder(self.embedding_size, self.embedding_size//2))
                print("Decoder", self.embedding_size, self.embedding_size//2)
            # halve the embedding size
            self.embedding_size = self.embedding_size // 2
        
        # save the decoder list as an nn.ModuleList to self.decoders
        
        

        # create a MaxPool2d of kernel size scale_factor as the final pooling layer
        self.pool = nn.MaxPool2d(kernel_size = (scale_factor, scale_factor))
        

    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        # evaluate x with self.inc
        x = self.doubleconvhelper(x)

        # create a list of the residuals, with its only element being x
        residuals = [x]
        # for each encoder
        for encoder in self.encoders:
            # run the residual through the encoder, append the output to the residual
            residuals.append(encoder(residuals[-1]))

        # set x to be the last value from the residuals
        x = residuals[-1] # Hmmmm missing a call to decoder.up
        
        # for each residual except the last one
        for i in range(len(residuals)-1):
            # Implement Attention Gate
            # evaluate it with the decoder
            # Issue: Would have to call activation inside decoder however we would need access to the prev residual
            # Split the decoder?
            prev_residual = self.activation_gates[len(self.activation_gates)-1-i](residuals[len(residuals)-2-i], x)
            x = self.decoders[i](x, prev_residual)
        
        # evaluate the final pooling layer
        x = self.pool(x)

        return x
        