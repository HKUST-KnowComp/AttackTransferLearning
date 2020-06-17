import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# For SVHN dataset
class DTN(nn.Module):
    '''
    named_params: 
    ['conv_params.0.weight',
     'conv_params.0.bias',
     'conv_params.1.weight',
     'conv_params.1.bias',
     'conv_params.4.weight',
     'conv_params.4.bias',
     'conv_params.5.weight',
     'conv_params.5.bias',
     'conv_params.8.weight',
     'conv_params.8.bias',
     'conv_params.9.weight',
     'conv_params.9.bias',
     'fc_params.0.weight',
     'fc_params.0.bias',
     'fc_params.1.weight',
     'fc_params.1.bias',
     'classifier.weight',
     'classifier.bias']
    '''
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

    def forward(self, x, return_feat=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        if return_feat:
            return x, F.log_softmax(y, dim=1)
        else:
            return F.log_softmax(y, dim=1)

    def extract_features(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x

    def get_parameters(self, init_lr):
        parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list

    def restore_from_ckpt(self, ckpt_state_dict, exclude_vars):
        '''restore entire model from another model
        Args:
            exclude_vars: a list of string, prefixes of variables to be excluded
        '''
        model_dict = self.state_dict()

        # Fiter out unneccessary keys
        print('restore variables: ')
        filtered_dict = {}
        for n, v in ckpt_state_dict.items():
            if len(exclude_vars) == 0:
                prefix_match = [0]
            else:
                prefix_match = [1 if n.startswith(vn) else 0 for vn in exclude_vars]
            if sum(prefix_match) == 0 and v.size() == model_dict[n].size():
                print(n)
                filtered_dict[n] = v
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        # frozen restored params
        print('freeze variables: ')
        if len(exclude_vars) > 0:
            for n,v in self.named_parameters():
                prefix_match = [1 if n.startswith(vn) else 0 for vn in exclude_vars]
                # if n is not found in exclude_vars, freeze it
                if sum(prefix_match) == 0:
                    v.requires_grad = False

# resnet for generic image classification
resnet_dict = {"ResNet18": models.resnet18,
               "ResNet34": models.resnet34,
               "ResNet50": models.resnet50,
               "ResNet101": models.resnet101,
               "ResNet152": models.resnet152
               }


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_feat=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if return_feat:
            return out, F.log_softmax(self.fc(out), dim=1)
        else:
            return F.log_softmax(self.fc(out), dim=1)

    def get_parameters(self, init_lr):
        parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list

    def restore_from_ckpt(self, ckpt_state_dict, exclude_vars):
        '''restore entire model from another model'''
        model_dict = self.state_dict()

        # Fiter out unneccessary keys
        filtered_dict = {}
        for n, v in ckpt_state_dict.items():
            if len(exclude_vars) == 0:
                prefix_match = [0]
            else:
                prefix_match = [1 if n.startswith(vn) else 0 for vn in exclude_vars]
            if sum(prefix_match) == 0 and v.size() == model_dict[n].size():
                print(n)
                filtered_dict[n] = v
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        # frozen restored params
        print('freeze variables: ')
        if len(exclude_vars) > 0:
            for n,v in self.named_parameters():
                prefix_match = [1 if n.startswith(vn) else 0 for vn in exclude_vars]
                # if n is not found in exclude_vars and n is not a var in bn layer, freeze it
                if sum(prefix_match) == 0 and 'bn' not in n:
                    print(n)
                    v.requires_grad = False
