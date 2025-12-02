from ...lib.result import Result
from .abstract import AbstractWifihdfPrivacy
from ...lib.inference import Classification

import copy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class BfidPrivacy(Classification, AbstractWifihdfPrivacy):
    """Use the LSTM model introduced by BFId: Identity Inference Attacks using Beamforming Feedback Information

    Required pips:
        - pytorch

    Parameters:
        None
    """

    def validate_config(self):
        for x in ['data_keys', 'input_size', 'hidden_size', 'rnn_layers', 'linear_size', 'epochs', 'lr', 'weight_decay', 'bs']:
            if x not in self.config:
                raise AttributeError("Configuration for BFID Classification is missing parameter {}!".format(x))

    def enroll(self, dataset):
        data = dataset.to_torch(self.config['data_keys'])
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=self.config['bs'], num_workers=4,pin_memory=True, shuffle=True, collate_fn=collate)
        loss_fn = torch.nn.CrossEntropyLoss()

        accs, losses = [], []
        epochs_since_last_improvement = 0

        self.device = ("cuda:0") if torch.cuda.is_available() else "cpu"
        self.model = SimpleLSTMBN(self.config['input_size'], len(dataset.identities), hidden_size=self.config['hidden_size'], rnn_layers=self.config['rnn_layers'], linear_size=self.config['linear_size'])
        self.model.to(self.device)
        self.idx_to_class = data.idx_to_class

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
        for x, l, y in dataloader:

            if len(x) == 0:
                continue
            else:
                processed_examples += len(x)
                steps_taken += 1

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            pred = self.model(x, l)
            loss = loss_fn(pred, y)

            total_acc += (pred.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)

            loss.backward()
            optimizer.step()

        return total_loss / processed_examples, total_acc / processed_examples

    def classify_all(self, dataset, resultset):
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()

        data = dataset.to_torch(self.config['data_keys'])
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=self.config['bs'], num_workers=4,pin_memory=True, shuffle=True, collate_fn=collate)

        with torch.no_grad():
            for x, l, y in dataloader:
                x = x.to(self.device)
                pred = self.model(x, l)
                for pre, real in zip(pred, y):
                    rs = Result(self.idx_to_class[real], 'unknown')
                    for i in range(len(pre)):
                        rs.add_recognized(self.idx_to_class[i], dist=1-pre[i])
                    resultset.append(rs)
        return resultset

def collate(batch):
    xs = []
    ys = []
    ls = []

    for x, y in batch:
        xs.append(x)
        ys.append(y)
        ls.append(len(x))

    padded_batch = pad_sequence(xs, batch_first=True)
    return padded_batch, torch.as_tensor(ls), torch.as_tensor(ys)

class SimpleLSTMBN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=160, rnn_layers=2, linear_size=112):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers,
                                 batch_first=True)
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_size, linear_size),
                                      torch.nn.BatchNorm1d(linear_size),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(linear_size, output_size),
                                      torch.nn.BatchNorm1d(output_size),
                                      torch.nn.ReLU()
                                      )

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l):
        out = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        out, (hn, cell) = self.rnn(out)
        out = hn[-1, :, :]
        out = self.fc(out)
        out = self.softmax(out)
        return out
