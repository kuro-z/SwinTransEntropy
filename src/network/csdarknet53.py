import torch.nn as nn
from src.network.cslayers import *
import torch.backends.cudnn as cudnn
from torchsummary import summary

__all__ = ['CsDarkNet53']

class CsDarkNet53(nn.Module):
    def __init__(self, input_channels):
        super(CsDarkNet53, self).__init__()

        # Network
        self.stage1 = Conv2dBatchLeaky(input_channels, input_channels, 3, 1)
        self.stage2 = Stage3(input_channels)
        self.stage3 = Stage3(input_channels)
        self.stage4 = Stage(input_channels, 8)

        self.conv = Conv2dBatchLeaky(input_channels, input_channels, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        conv = self.conv(stage4)
        return conv

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    darknet = CsDarkNet53(num_classes=10)
    darknet = darknet.cuda()
    with torch.no_grad():
        darknet.eval()
        data = torch.rand(1, 960, 16, 16)
        data = data.cuda()
        try:
            #print(darknet)
            summary(darknet,(960,16,16))
            print(darknet(data))
        except Exception as e:
            print(e)
