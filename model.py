import torch
import torch.nn as nn
import torch.nn.functional as F

# import CapsNet
# import OctConv
# from PASE.pase.models.frontend import wf_builder

from PIL import Image


class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv6a = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=4)
        self.conv6b = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=1)
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=80, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # GAP
        # x = self.gap(x)
        # x=x.squeeze()
        # x = self.fc(x)

        # attention
        xa = self.conv6a(x)
        xb = self.conv6b(x)
        x = xa.mul(xb)
        x = self.gap(x)
        x = x.squeeze()
        if (len(x.shape) < 2):
            x = x.unsqueeze(0)

        return x


class VLSS_CNN(nn.Module):
    def __init__(self):
        super(VLSS_CNN, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size=(1, 12), in_channels=1, out_channels=6)
        self.maxp = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(kernel_size=(1, 8), in_channels=6, out_channels=10)
        self.gru = nn.GRU(input_size=1152, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(2560, 4)

    def forward(self, *input):
        x = input[0]
        x = self.conv1(x)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.maxp(x)
        x = x.view(x.size()[0], x.size()[1], x.size()[2] * x.size()[3])
        x, _ = self.gru(x)
        x = x.contiguous().view(x.size()[0], x.size()[1] * x.size()[2])
        x = self.dense(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.softmax(x)
        return x


# class CapNetCNN(nn.Module):
#     def __init__(self, Lambda=0.5):
#         super(CapNetCNN, self).__init__()
#         self.Lambda = Lambda
#         self.conv1a = nn.Conv2d(kernel_size=(8, 2), in_channels=1, out_channels=8, padding=(3, 0))
#         self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
#         self.maxp1 = nn.MaxPool2d(kernel_size=(2, 1))
#         self.conv2 = nn.Conv2d(kernel_size=(5, 5), in_channels=16, out_channels=16, padding=2)
#         self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv3 = nn.Conv2d(kernel_size=(5, 5), in_channels=16, out_channels=16, padding=2)
#         self.maxp3 = nn.MaxPool2d(kernel_size=(4, 1))
#
#         self.bigru = nn.GRU(input_size=96, hidden_size=64, bidirectional=True, batch_first=True)
#         self.fc1 = nn.Linear(1152, 64)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(64, 4)
#
#         self.cap = CapsNet.CapsuleNet(input_size=[16, 6, 6], classes=4, routings=4)
#
#     def forward(self, *input):
#         x = input[0]
#         xa = self.conv1a(x)
#         xb = self.conv1b(x)
#         x = torch.cat((xa, xb), 1)
#         x = self.maxp1(x)
#         x = self.conv2(x)
#         x = self.maxp2(x)
#         x = self.conv3(x)
#         xc, _ = self.cap(x)
#         x = self.maxp2(x)
#         x = self.maxp3(x)
#         x = x.contiguous().view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3])
#         x = x.permute(0, 2, 1)
#         x, _ = self.bigru(x)
#         x = x.contiguous().view(x.size()[0], x.size()[1] * x.size()[2])
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = F.softmax(x)
#         out = xc * self.Lambda + x * (1 - self.Lambda)
#         return out


# class OctConvACNN(nn.Module):
#     def __init__(self):
#         super(OctConvACNN, self).__init__()
#         self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
#         self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
#         self.conv2 = OctConv.OctConv(kernel_size=3, ch_in=16, ch_out=32, alphas=(0, 0.5))
#         self.conv3 = OctConv.OctConv(kernel_size=3, ch_in=32, ch_out=48)
#         self.conv4 = OctConv.OctConv(kernel_size=3, ch_in=48, ch_out=64)
#         self.conv5 = OctConv.OctConv(kernel_size=3, ch_in=64, ch_out=80, alphas=(0.5, 0))
#         self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv6a = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=4)
#         self.conv6b = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=1)
#         self.bn1a = nn.BatchNorm2d(8)
#         self.bn1b = nn.BatchNorm2d(8)
#         self.bn2h = nn.BatchNorm2d(16)
#         self.bn2l = nn.BatchNorm2d(16)
#         self.bn3h = nn.BatchNorm2d(24)
#         self.bn3l = nn.BatchNorm2d(24)
#         self.bn4h = nn.BatchNorm2d(32)
#         self.bn4l = nn.BatchNorm2d(32)
#         self.bn5 = nn.BatchNorm2d(80)
#
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, *input):
#         xa = self.conv1a(input[0])
#         xa = self.bn1a(xa)
#         xa = F.relu(xa)
#         xb = self.conv1b(input[0])
#         xb = self.bn1b(xb)
#         xb = F.relu(xb)
#         x = torch.cat((xa, xb), 1)
#         x = self.conv2(x)
#
#         x = self.bn2h(x[0]), self.bn2l(x[1])
#         x = F.relu(x[0]), F.relu(x[1])
#         x = self.maxp(x[0]), self.maxp(x[1])
#
#         x = self.conv3(x)
#         x = self.bn3h(x[0]), self.bn3l(x[1])
#         x = F.relu(x[0]), F.relu(x[1])
#         x = self.maxp(x[0]), self.maxp(x[1])
#
#         x = self.conv4(x)
#         x = self.bn4h(x[0]), self.bn4l(x[1])
#         x = F.relu(x[0]), F.relu(x[1])
#
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = F.relu(x)
#
#         # GAP
#         # x = self.gap(x)
#         # x=x.squeeze()
#         # x = self.fc(x)
#
#         # attention
#         xa = self.conv6a(x)
#         xb = self.conv6b(x)
#         x = xa.mul(xb)
#         x = self.gap(x)
#         x = x.squeeze()
#
#         x = F.softmax(x, dim=1)
#         return x



# class PaseACNN(nn.Module):
#     def __init__(self):
#         super(PaseACNN, self).__init__()
#         self.conv1 = nn.Conv1d(kernel_size=3, in_channels=132, out_channels=128, padding=1)
#         self.conv2 = nn.Conv1d(kernel_size=3, in_channels=128, out_channels=144, padding=1)
#         self.maxp = nn.MaxPool1d(kernel_size=2)
#         self.conv3a = nn.Conv1d(kernel_size=1, in_channels=144, out_channels=4)
#         self.conv3b = nn.Conv1d(kernel_size=1, in_channels=144, out_channels=1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(144)
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(in_features=144, out_features=4)
#
#         self.pase = wf_builder('PASE/cfg/PASE.cfg')
#         # self.pase.eval()
#         self.pase.load_pretrained('PASE/PASE.ckpt', load_last=True, verbose=True)
#         # self.pase.load_pretrained('E:\Test/FE_e72.ckpt', load_last=True, verbose=True)
#
#         self.conx = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=32)
#         self.avgp = nn.MaxPool1d(160)
#
#     def forward(self, *input):
#         x = input[0]
#         # x = x.unsqueeze(1)
#         # self.pase.eval()
#         x = self.pase(x)
#         x2 = self.conx(input[0])
#         x2 = self.avgp(x2)
#         x = torch.cat((x, x2), dim=1)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.maxp(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.maxp(x)
#
#         # attention
#         xa = self.conv3a(x)
#         xb = self.conv3b(x)
#         x = xa.mul(xb)
#         x = self.gap(x)
#         x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
#         x = F.softmax(x, dim=1)
#
#         return x


class MACNN(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(MACNN, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=self.attention_hidden, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))


    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        # #attention

        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x
