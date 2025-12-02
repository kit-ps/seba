from ...lib.result import Result
from .abstract import AbstractWifihdfPrivacy
from ...lib.inference import Classification
from ...splitter.withinlabel1to2 import Withinlabel1to2Splitter

import torch
import copy

class CautionPrivacy(Classification, AbstractWifihdfPrivacy):
    """Uses the model by Wang et al. from "CAUTION: A Robust WiFi-based Human Authentication System via Few-shot Open-set Recognition"

    Required pips:
        - pytorch

    Parameters:
        None
    """

    def validate_config(self):
        for x in ['lr', 'weight_decay', 'epochs', 'early_stopping', 'bs']:
            if x not in self.config:
                raise AttributeError("Configuration for Caution Classification is missing parameter {}!".format(x))

    def enroll(self, dataset):
        # make data length the same for all and remove timestamp-delta
        dataset.cut_or_pad(self.config['length'])
        dataset.remove_ts()
        self.enrollset = dataset
        self.idx_to_class = dataset.identities

        # split dataset into support and query
        splitter = Withinlabel1to2Splitter({'rate': 0.5}, 'support|query', None)
        support_ds, query_ds = splitter.run([dataset])

        query_dl = torch.utils.data.DataLoader(
            dataset=query_ds.to_torch(['flattened_mags'], normalize=True, ts_removed=True),
            batch_size=self.config['bs'], num_workers=4, pin_memory=True, shuffle=True
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        accs, losses = [], []
        epochs_since_last_improvement = 0

        self.device = ("cuda:0") if torch.cuda.is_available() else "cpu"
        self.model = Caution({'channel_in': 2, 'hid_layer': self.config['hid_layer'], 'encoding_dim': 512, 'shape_in': (1, 2, self.config['length'], 53)})
        self.model.device = self.device
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epochs']):
            centers = self.calculate_centers(support_ds)
            loss, acc = self.train_model(query_dl, centers, optimizer, loss_fn)
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

    def train_model(self, dl, centers, optimizer, loss_fn):
        self.model.train()
        total_acc, total_loss = 0, 0
        processed_examples, steps_taken = 0, 0
        for x, y in dl:

            if len(x) == 0:
                continue
            else:
                processed_examples += len(x)
                steps_taken += 1

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            enc = self.model.encoder(x)
            pred = self.model.classify(enc, centers)
            loss = loss_fn(pred, y)

            total_acc += (pred.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)

            loss.backward()
            optimizer.step()

        return total_loss / processed_examples, total_acc / processed_examples

    @torch.no_grad()
    def calculate_centers(self, support_ds):
        self.model.eval()
        # some magic to make sure we have one batch per label
        ds = support_ds.to_torch(['flattened_mags'], normalize=True, ts_removed=True)
        class_inds = [torch.where(torch.tensor(list(map(lambda x: ds.class_to_idx[x], ds.y))) == class_idx)[0]
                      for class_idx in ds.class_to_idx.values()]
        dataloaders = [
            torch.utils.data.DataLoader(
                dataset=torch.utils.data.Subset(ds, inds),
                batch_size=max(map(lambda x: len(x), class_inds)),
                pin_memory=True,
                shuffle=True,
                drop_last=False)
            for inds in class_inds
        ]

        centers = torch.zeros((len(ds.class_to_idx), 512), device=self.device)
        for dl in dataloaders:
            for values, labels in dl:
                centers[labels[0], :] = self.model.encoder(values.to(self.device)).mean(dim = 0)
        return centers

    def classify_all(self, dataset, resultset):
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()

        centers = self.calculate_centers(self.enrollset)
        dataset.cut_or_pad(self.config['length'])
        dataset.remove_ts()
        data = dataset.to_torch(['flattened_mags'], normalize=True, ts_removed=True)
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=self.config['bs'], num_workers=4, pin_memory=True, shuffle=True)

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                enc = self.model.encoder(x)
                pred = self.model.classify(enc, centers)
                for pre, real in zip(pred, y):
                    rs = Result(self.idx_to_class[real], 'unknown')
                    for i in range(len(pre)):
                        rs.add_recognized(self.idx_to_class[i], dist=1-pre[i])
                    resultset.append(rs)
        return resultset


# FROM https://github.com/DarriusL/IDLab/blob/main/model/framework/caution.py
class Caution(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__();

        #input:[B, 6000, 3, 56] -> [B, 3, 6000, 56] -> [B, 128, 750, 7]
        #128 = channels[-1]
        channel_in = config['channel_in']
        channels = config['hid_layer'];
        assert len(channels) == 3
        activation_fn = torch.nn.LeakyReLU();
        layers = [
            torch.nn.Conv2d(channel_in, channels[0], kernel_size = (5, 5), padding = 2),
            activation_fn,
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ] + [
            layer
            for i in range(len(channels) - 1)
            for layer in [
            torch.nn.Conv2d(channels[i], channels[i + 1], kernel_size=(5, 5), padding=2),
            activation_fn,
            torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        ];
        self.cnn = torch.nn.Sequential(*layers)
        lin_in_shape = self.cnn(torch.rand(*(config['shape_in']))).data.reshape(1, -1).shape[1]
        self.feature = torch.nn.Linear(lin_in_shape, config['encoding_dim'])

    def encoder(self, amps):
        #amps:[B, 6000, 3, 56] -> [B, 3, 6000, 56]
        feature1 = self.cnn(amps.reshape(amps.shape[0], amps.shape[1], 2, -1).permute(0, 2, 1, 3))
        feature2 = self.feature(feature1.reshape(amps.shape[0], -1))
        #[B, do]
        return feature2

    def classify(self, encs, centers):
        distances = torch.exp( - torch.cdist(encs, centers))
        probs = distances / distances.sum(dim = 1, keepdim = True)
        return probs
