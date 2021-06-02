import os
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules import resnet_block

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, stride=2, dimension=D)

        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D)

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)

        self.conv5 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)

        self.glob_avg = ME.MinkowskiGlobalMaxPooling(dimension=D)

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    dimension=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.glob_avg(x)
        x = self.final(x)
        return x


class ResNet18(ResNetBase):
    BLOCK = resnet_block.BasicBlock
    LAYERS = (2, 2, 2, 2)


class SparseResNet18(nn.Module):
    def __init__(self, fc_dim=64, in_channels=3, out_channels=40, D=3):
        super(SparseResNet18, self).__init__()
        original_resnet = ResNet18(in_channels=in_channels, out_channels=out_channels, D=D)
        checkpoint_path = os.path.join(os.path.split(os.path.join(os.getcwd(), __file__))[0], 'resnet18_pretrained.pth')

        #Load modelnet40 pretrained weights
        checkpoint = torch.load(checkpoint_path)
        original_resnet.load_state_dict(checkpoint['state_dict'])

        self.features = nn.Sequential(*list(original_resnet.children())[:-4])

        self.fc = ME.MinkowskiConvolution(in_channels=512, out_channels=fc_dim, kernel_size=3, stride=1, dimension=D)
        self.glob_avg = ME.MinkowskiGlobalMaxPooling(dimension=D)

    def forward(self, x, pool=True):

        x = self.features(x) #input is ME.SparseTensor
        x = self.fc(x)
        x = self.glob_avg(x)
        return x.F #output is torch.Tensor
