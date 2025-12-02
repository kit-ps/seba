from ...lib.result import Result
from .abstract import AbstractWifihdfPrivacy
from ...lib.inference import Classification
from ...dataset.wifihdf import SingleTorchWifiDataset

import torch
import copy

class LwwiidPrivacy(Classification, AbstractWifihdfPrivacy):
    """Uses the model by Cao et al. from "A Lightweight Depe Learning Algorithm for WiFi-Based Identity Recognition"

    Required pips:
        - pytorch

    Parameters:
        None
    """

    def validate_config(self):
        for x in ['sliding_step', 'window_size', 'lr', 'weight_decay', 'epochs', 'early_stopping']:
            if x not in self.config:
                raise AttributeError("Configuration for LW-WiiD Classification is missing parameter {}!".format(x))

    def enroll(self, dataset):
        data = self.to_feg_data(dataset)
        self.idx_to_class = data.idx_to_class
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=self.config['bs'], num_workers=4,pin_memory=True, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()

        accs, losses = [], []
        epochs_since_last_improvement = 0

        self.device = ("cuda:0") if torch.cuda.is_available() else "cpu"
        self.model = WiAU(len(dataset.identities))
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epochs']):
            loss, acc = self.train_model(dataloader, optimizer, loss_fn)
            accs.append(acc)
            losses.append(loss)
            self.log.info("Training Epoch {}: Accuracy {}, Loss {}".format(epoch, acc, loss))
            if len(losses) == 1 or loss < min(losses[:-1]):
                epochs_since_last_improvement = -1
            if len(accs) == 1 or acc > max(accs[:-1]):
                epochs_since_last_improvement = -1
                self.best_model_dict = copy.deepcopy(self.model.state_dict())
            epochs_since_last_improvement += 1
            if self.config['early_stopping']:
                if epochs_since_last_improvement > self.config['early_stopping']:
                    self.log.info("Early stopping training.")
                    break

    def train_model(self, dataloader, optimizer, loss_fn):
        self.model.train()
        total_acc, total_loss = 0, 0
        processed_examples, steps_taken = 0, 0
        for x, y in dataloader:

            if len(x) == 0:
                continue
            else:
                processed_examples += len(x)
                steps_taken += 1

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            pred = self.model(x)
            loss = loss_fn(pred, y)

            total_acc += (pred.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)

            loss.backward()
            optimizer.step()

        return total_loss / processed_examples, total_acc / processed_examples

    def classify_all(self, dataset, resultset):
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()

        data = self.to_feg_data(dataset)
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=self.config['bs'], num_workers=4,pin_memory=True, shuffle=True)

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                pred = self.model(x)
                for pre, real in zip(pred, y):
                    rs = Result(self.idx_to_class[real], 'unknown')
                    for i in range(len(pre)):
                        rs.add_recognized(self.idx_to_class[i], dist=1-pre[i])
                    resultset.append(rs)
        return resultset

    def to_feg_data(self, dataset):
        sliding_step = self.config['sliding_step']
        window_size = self.config['window_size']
        data = []
        labels = []
        orig = dataset.to_torch(['flattened_mags'], normalize=False)
        for sample, label in orig:
            for i in range(((len(sample) - window_size) // sliding_step) + 1):
                data.append(torch.stack([
                    sample[(i*sliding_step):(i*sliding_step+window_size), :(len(sample[0])-1)//2].transpose(0, 1),
                    sample[(i*sliding_step):(i*sliding_step+window_size), (len(sample[0])-1)//2:-1].transpose(0, 1)
                ]))
                labels.append(orig.idx_to_class[label])
        return SingleTorchWifiDataset(data, labels, orig.class_to_idx, orig.idx_to_class)


# FROM https://github.com/DarriusL/IDLab/blob/main/model/framework/wiau.py
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ResNetUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * 4)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class WiAU(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.cnn_unit = ConvBlock(2, 64, kernel_size=7, stride=2, padding=3)

        self.resnet_layers = torch.nn.Sequential(
            ResNetUnit(64, 32, stride=1),
            ResNetUnit(128, 32, stride=1),
            ResNetUnit(128, 64, stride=2),
            ResNetUnit(256, 128, stride=2),
            ResNetUnit(512, 256, stride=2)
        )

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_identity = torch.nn.Linear(1024, output_size);

    def forward(self, x):
        x = self.cnn_unit(x)
        x = self.resnet_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        identity_out = self.fc_identity(x)
        return identity_out
