
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv



class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r
        
    def forward(self, input_tensor):
        """
        :param input_tensor: shape (batch, channels, x_dim, y_dim)
        
        """
        batch_dim, channel_dim, x_dim, y_dim = input_tensor.shape
        
        xx_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)

        xx_range = torch.arange(x_dim, dtype=torch.int32)
        yy_range = torch.arange(y_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_dim, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_dim, 1, 1, 1)

        if torch.cuda.is_available:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()

        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)
                    
        return out

    
class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        
        self.addcoords = AddCoords(with_r)
        # CALL A REGULAR CONV WITH THE TOTAL NUMBER OF CHANNELS
        self.conv = nn.Conv2d(in_channels + 2 + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv(nn.Module):
    def __init__(self, canvas_size=28, nbr_channels=1):
        super(CoordConv, self).__init__()
        self.nbr_channels = nbr_channels
        self.canvas_size = canvas_size
        self.coordconv = CoordConv2d(self.nbr_channels, 32, 1, with_r=False)
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64,  1, 1)
        self.conv4 = nn.Conv2d( 1,  1, 1)
        #self.sofmax2D = SoftmaxLogProbability2D()  #  2D sofmax
        #self.transformer = STNet()  #  Spatial transformer

    def forward(self, x):
        x = self.coordconv(x)    # Coordconv
        #x = self.transformer(x)  # Transformer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        #x = self.sofmax2D(x)   # 2D softmax
        x = x.view(-1, self.canvas_size*self.canvas_size)
        return x


class STCoordConv(nn.Module):
    def __init__(self, canvas_size=28, nbr_channels=1):
        super(STCoordConv, self).__init__()
        self.nbr_channels = nbr_channels
        self.canvas_size = canvas_size
        self.coordconv = CoordConv2d(self.nbr_channels, 32, 1, with_r=False)
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64,  1, 1)
        self.conv4 = nn.Conv2d( 1,  1, 1)
        self.sofmax2D = SoftmaxLogProbability2D()  #  2D sofmax
        self.transformer = STNet()  #  Spatial transformer

    def forward(self, x):
        x = self.coordconv(x)    # Coordconv
        x = self.transformer(x)  # Transformer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.sofmax2D(x)   # 2D softmax
        x = x.view(-1, self.canvas_size*self.canvas_size)
        return x


class SoftmaxLogProbability2D(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLogProbability2D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2] * orig_shape[3])), dim=1)\
                .view((orig_shape[0], orig_shape[2], orig_shape[3]))
            seq_x.append(softmax_.log())
        x = torch.stack(seq_x, dim=1)
        return x
    

class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x 
    