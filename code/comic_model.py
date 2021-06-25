import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET

from typing import Callable, Any, Optional, Tuple, List

# class CNN_ENCODER(nn.Module):
#     def __init__(self, nef):
#         super(CNN_ENCODER, self).__init__()
#         if cfg.TRAIN.FLAG:
#             self.nef = nef
#         else:
#             self.nef = 256  # define a uniform ranker
#
#         model = models.inception_v3()
#         url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
#         model.load_state_dict(model_zoo.load_url(url))
#         for param in model.parameters():
#             param.requires_grad = False
#         print('Load pretrained model from ', url)
#         # print(model)
#
#         self.define_module(model)
#         self.init_trainable_weights()
#
#     def define_module(self, model):
#         self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
#         self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
#         self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
#         self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
#         self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
#         self.Mixed_5b = model.Mixed_5b
#         self.Mixed_5c = model.Mixed_5c
#         self.Mixed_5d = model.Mixed_5d
#         self.Mixed_6a = model.Mixed_6a
#         self.Mixed_6b = model.Mixed_6b
#         self.Mixed_6c = model.Mixed_6c
#         self.Mixed_6d = model.Mixed_6d
#         self.Mixed_6e = model.Mixed_6e
#         self.Mixed_7a = model.Mixed_7a
#         self.Mixed_7b = model.Mixed_7b
#         self.Mixed_7c = model.Mixed_7c
#
#         self.emb_features = conv1x1(768, self.nef)
#         self.emb_cnn_code = nn.Linear(2048, self.nef)
#
#     def init_trainable_weights(self):
#         initrange = 0.1
#         self.emb_features.weight.data.uniform_(-initrange, initrange)
#         self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, x):
#         features = None
#         # --> fixed-size input: batch x 3 x 299 x 299
#         x = nn.functional.interpolate(x,size=(299, 299), mode='bilinear', align_corners=False)
#         # 299 x 299 x 3
#         x = self.Conv2d_1a_3x3(x)
#         # 149 x 149 x 32
#         x = self.Conv2d_2a_3x3(x)
#         # 147 x 147 x 32
#         x = self.Conv2d_2b_3x3(x)
#         # 147 x 147 x 64
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 73 x 73 x 64
#         x = self.Conv2d_3b_1x1(x)
#         # 73 x 73 x 80
#         x = self.Conv2d_4a_3x3(x)
#         # 71 x 71 x 192
#
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 35 x 35 x 192
#         x = self.Mixed_5b(x)
#         # 35 x 35 x 256
#         x = self.Mixed_5c(x)
#         # 35 x 35 x 288
#         x = self.Mixed_5d(x)
#         # 35 x 35 x 288
#
#         x = self.Mixed_6a(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6b(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6c(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6d(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6e(x)
#         # 17 x 17 x 768
#
#         # image region features
#         features = x
#         # 17 x 17 x 768
#
#         x = self.Mixed_7a(x)
#         # 8 x 8 x 1280
#         x = self.Mixed_7b(x)
#         # 8 x 8 x 2048
#         x = self.Mixed_7c(x)
#         # 8 x 8 x 2048
#         x = F.avg_pool2d(x, kernel_size=8)
#         # 1 x 1 x 2048
#         # x = F.dropout(x, training=self.training)
#         # 1 x 1 x 2048
#         x = x.view(x.size(0), -1)
#         # 2048
#
#         # global image features
#         cnn_code = self.emb_cnn_code(x)
#         # 512
#         if features is not None:
#             features = self.emb_features(features)
#         return features, cnn_code

# --------------------------------------------------------------------------------------------------------------------


class ComicClassifier(nn.Module):

    def __init__(self, num_classes):
        super(ComicClassifier, self).__init__()

        conv_block = BasicConv2d
        inception_a = CustomInceptionA
        inception_b = CustomInceptionB
        inception_c = CustomInceptionC
        inception_aux = CustomInceptionAux
        inception_d = CustomInceptionD
        inception_e = CustomInceptionE

        # 32 to 16
        # N x 3 x 299 x 299
        self.Conv1 = conv_block(3, 16, kernel_size=3, stride=2)
        # 64 to 32
        # N x 16 x 149 x 149
        self.Conv2 = conv_block(16, 32, kernel_size=3)
        # N x 32 x 147 x 147

        self.Maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # N x 32 x 73 x 73

        self.Conv3 = conv_block(32, 40, kernel_size=1)
        # N x 40 x 73 x 73
        self.Conv3b = conv_block(40, 96, kernel_size=3)
        # 96 x 71 x 71

        self.Maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # N x 96 x 35 x 35

        # custom inception a

        self.Mixed5b = inception_a(96, pool_features=16)
        # N x 128 x 35 x 35

        # custom inception b
        self.Mixed6a = inception_b(128)
        # N x 320 x 17 x 17

        # custom inception c
        # self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed6b = inception_c(320, channels_7x7=64)
        # N x 320 x 17 x 17

        # features are here 17x17 regions

        self.AuxLogits = inception_aux(320, num_classes)

        self.Mixed7a = inception_d(320)
        # 500 x 8 x 8

        self.mixed7b = inception_e(500)
        # 640 x 8 x 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(960, num_classes)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor):

        # N x 3 x 299 x 299
        x = self.Conv1(x)
        # N x 16 x 149 x 149
        x = self.Conv2(x)
        # N x 32 x 147 x 147
        x = self.Maxpool1(x)
        # N x 32 x 73 x 73
        x = self.Conv3(x)
        # N x 40 x 73 x 73
        x = self.Conv3b(x)
        # 96 x 71 x 71
        x = self.Maxpool2(x)
        # N x 96 x 35 x 35
        # custom inception a
        x = self.Mixed5b(x)
        # N x 128 x 35 x 35
        # custom inception b
        x = self.Mixed6a(x)
        # N x 320 x 17 x 17
        # custom inception c
        x = self.Mixed6b(x)
        # N x 320 x 17 x 17

        # features are here 17x17 regions
        aux = self.AuxLogits(x)

        # custom inception d
        x = self.Mixed7a(x)
        # 500 x 8 x 8
        # custom inception e
        x = self.Mixed7b(x)
        # 640 x 8 x 8
        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        prediction = self.fc(x)

        return prediction, aux

    def forward(self, x:Tensor):
        x = self._transform_input(x)

        x, aux = self._forward(x)

        if self.training:
            return x, aux

        else:
            return x


class COMIC_CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(COMIC_CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        self.transform_input = True

        self.base_model = ComicClassifier(num_classes=13)

        path = "comic_classifier.pt"

        self.base_model.load_state_dict(torch.load(path))
        for param in self.base_model.parameters():
            param.requires_grad = False

        print('Load pretrained comics cnn model')

        self.define_module(self.base_model)
        self.init_trainable_weights()


    def define_module(self, model):
        self.Conv1 = self.base_model.Conv1
        self.Conv2 = self.base_model.Conv2
        self.Maxpool1 = self.base_model.Maxpool1
        self.Conv3 = self.base_model.Conv3
        self.Conv3b = self.base_model.Conv3b
        self.Maxpool2 = self.base_model.Maxpool2
        self.Mixed5b = self.base_model.Mixed5b
        self.Mixed6a = self.base_model.Mixed6a
        self.Mixed6b = self.base_model.Mixed6b
        self.Mixed7a = self.base_model.Mixed7a
        self.Mixed7b = self.base_model.mixed7b

        self.avgpool = self.base_model.avgpool
        self.dropout = self.base_model.dropout

        self.predict = self.base_model.fc

        self.emb_features = conv1x1(320, self.nef)
        self.emb_cnn_code = nn.Linear(960, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x


    def _forward(self, x: Tensor):
        # N x 3 x 299 x 299
        x = self.Conv1(x)
        # N x 16 x 149 x 149
        x = self.Conv2(x)
        # N x 32 x 147 x 147
        x = self.Maxpool1(x)
        # N x 32 x 73 x 73
        x = self.Conv3(x)
        # N x 40 x 73 x 73
        x = self.Conv3b(x)
        # 96 x 71 x 71
        x = self.Maxpool2(x)
        # N x 96 x 35 x 35
        # custom inception a
        x = self.Mixed5b(x)
        # N x 128 x 35 x 35
        # custom inception b
        x = self.Mixed6a(x)
        # N x 320 x 17 x 17
        # custom inception c
        x = self.Mixed6b(x)
        # N x 320 x 17 x 17

        features = x
        # features are here 17x17 regions
        # aux = self.AuxLogits(x)

        # custom inception d
        x = self.Mixed7a(x)
        # 500 x 8 x 8
        # custom inception e
        x = self.Mixed7b(x)
        # 960 x 8 x 8
        x = self.avgpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        # prediction = self.predict(x)

        cnn_code = self.emb_cnn_code(x)
        #         # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


    def forward(self, x):
        x = self._transform_input(x)

        features, cnn_code = self._forward(x)

        return features, cnn_code



class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class CustomInceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int = 16,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomInceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 32, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 24, kernel_size=1)
        self.branch5x5_2 = conv_block(24, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class CustomInceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomInceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 144, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        # 144 + 48 + 128
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class CustomInceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomInceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 80, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 80, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 80, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 80, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class CustomInceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomInceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = conv_block(64, 100, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch7x7x3_2 = conv_block(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(64, 64, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(64, 80, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]

        # 100 + 80 + 320
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class CustomInceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int = 500,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomInceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 160, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 160, kernel_size=1)
        self.branch3x3_2a = conv_block(160, 160, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(160, 160, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 200, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(200, 160, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(160, 160, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(160, 160, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 160, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        # 160 + 160 + 160 + 160

        # 640
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


# class CNN_IMAGE_ENCODER_SQUEEZENET(nn.Module):

class CustomInceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomInceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 40, kernel_size=1)
        self.conv1 = conv_block(40, 320, kernel_size=5)
        self.conv1.stddev = 0.01  # typeignore[assignment]
        self.fc = nn.Linear(320, num_classes)
        self.fc.stddev = 0.001  # typeignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 320 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 320 x 5 x 5
        x = self.conv0(x)
        # N x 40 x 5 x 5
        x = self.conv1(x)
        # N x 320 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 320 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 320
        x = self.fc(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

