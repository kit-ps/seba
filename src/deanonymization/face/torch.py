from .abstract import AbstractFaceDeanonymization
from ...lib.torch.data import TorchImageDataset, TupleDataset

import torch
import torch.nn as nn

import os
import uuid
import copy


class TorchDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by training a machine learning model on anonymized and corresponding clear images.

    Requires a specific machine learning model - this is the abstract class.
    Use convae, convlinae, or feedforward instead.

    Required pips:
        - torch
        - opencv-python
        - numpy

    Parameters:
        none
    """

    name = "torch"

    def validate_config(self):
        if "epochs" not in self.config:
            raise AttributeError("Torch Deanonymization requires number of epochs")
        if "train_rate" not in self.config:
            raise AttributeError("Torch Deanonymization requires train rate")
        if "learning_rate" not in self.config:
            raise AttributeError("Torch Deanonymization requires learning rate")
        if "train_batch_size" not in self.config:
            raise AttributeError("Torch Deanonymization requires training batch size")
        if "validation_batch_size" not in self.config:
            raise AttributeError("Torch Deanonymization requires validation batch size")
        if "loss" not in self.config:
            raise AttributeError("Torch Deanonymization requires loss function")

        if "weight_decay" not in self.config:
            self.config["weight_decay"] = 0.0
        if "early_stop" not in self.config:
            self.config["early_stop"] = 20
        if "reduce_lr" not in self.config:
            self.config["reduce_lr"] = True

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self, clear_set, anon_set):
        if self.load_model(clear_set):
            return

        self.load_data(clear_set, anon_set)

        self.create_model()
        self.log.debug("Created model with {} trainable parameters.".format(self.count_parameters(self.model)))
        self.load_loss_function()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.model.input = self.input_dim

        scheduler = None
        if self.config["reduce_lr"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=0.75, patience=5)

        cost_list_training = []
        cost_list_validation = []
        last_update = 0

        for epoch in range(self.config["epochs"]):
            cost_training = 0
            for anon, clear in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()

                output = self.model(anon.to(device).view([-1] + self.input_dim))
                loss = self.criterion(output.view(self.loss_input_shape), clear.to(device).view(self.loss_input_shape))
                loss.backward()
                self.optimizer.step()
                cost_training += loss.item()
            cost_list_training.append(cost_training)

            cost_val = 0
            cost_val2 = 0
            for anon, clear in self.validation_loader:
                self.model.eval()
                output = self.model(anon.to(device).view([-1] + self.input_dim))
                loss = self.criterion(output.view(self.loss_input_shape), clear.to(device).view(self.loss_input_shape))
                cost_val += loss.item()
                if self.c2:
                    loss2 = self.crit2(output.view(self.lis2), clear.to(device).view(self.lis2))
                    cost_val2 += loss2.item()

            cost_list_validation.append(cost_val)
            self.log.info(
                "epoch {}, cost-train {}, cost-val {}, cost-val-ssim {}, lr {}".format(
                    epoch, cost_training, cost_val, cost_val2, self.optimizer.param_groups[0]["lr"]
                )
            )

            if len(cost_list_validation) == 1 or cost_val <= min(cost_list_validation[:-1]) - 1e-05:
                self.save_model(clear_set)
                self.log.info("found new minimum validation cost => saved model!")
                last_update = 0
            else:
                last_update += 1
                if last_update >= self.config["early_stop"]:
                    self.log.info("no new minimum for {} epochs, stopping early".format(self.config["early_stop"]))
                    break

            if self.config["reduce_lr"]:
                scheduler.step(cost_val)

        self.save_model(clear_set, suffix="final")
        self.load_model(clear_set)  # Reload best model (last might not be best)

    def load_model(self, clear_set):
        clear_set.reload_meta()
        if "models" not in clear_set.meta:
            clear_set.meta["models"] = []

        for i in range(len(clear_set.meta["models"])):
            x = clear_set.meta["models"][i]
            if x["name"] == self.name and x["suffix"] == "none" and x["config"] == self.config:
                if os.path.exists(x["path"]):
                    self.model = torch.load(x["path"])
                    if not hasattr(self.model, "resize"):
                        self.model.resize = False
                    self.log.info("Using existing model: " + x["path"])
                    return True
        return False

    def load_data(self, clear_set, anon_set):
        clear = TorchImageDataset.from_set(clear_set)
        anon = TorchImageDataset.from_set(anon_set)
        self.input_shape = anon[0].shape
        self.output_shape = clear[0].shape

        set = TupleDataset(anon, clear)
        set.shuffle()
        train_dataset, validation_dataset = set.split(self.config["train_rate"])

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config["train_batch_size"], shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=self.config["validation_batch_size"])

    def load_loss_function(self):
        self.c2 = False
        if self.config["loss"] == "mse":
            self.loss_input_shape = [-1] + self.input_dim
            self.criterion = nn.MSELoss()

        elif self.config["loss"] == "mae":
            self.loss_input_shape = [-1] + self.input_dim
            self.criterion = nn.L1Loss()

        elif self.config["loss"] == "arcface":
            from ...lib.torch.arcface import ArcFace_Loss

            self.loss_input_shape = [-1] + list(self.output_shape)
            self.criterion = ArcFace_Loss()

            from ...lib.torch.ssim import SSIM_Loss

            self.c2 = True
            self.lis2 = [-1] + list(self.output_shape)
            self.crit2 = SSIM_Loss(data_range=1.0, channel=self.output_shape[0])

        elif self.config["loss"] == "ssim":
            from ...lib.torch.ssim import SSIM_Loss

            self.loss_input_shape = [-1] + list(self.output_shape)
            self.criterion = SSIM_Loss(data_range=1.0, channel=self.output_shape[0])

        elif self.config["loss"] == "msssim":
            from ...lib.torch.ssim import MSSSIM_Loss

            self.loss_input_shape = [-1] + list(self.output_shape)
            self.criterion = MSSSIM_Loss(data_range=1.0, channel=self.output_shape[0])

        else:
            raise ValueError("unknown loss function " + self.config["loss"])

    def save_model(self, clear_set, suffix="none"):
        model_path = os.path.join(clear_set.folder, "model-" + str(uuid.uuid4())[:8] + ".pth")
        torch.save(self.model, model_path)

        clear_set.reload_meta()
        if "models" not in clear_set.meta:
            clear_set.meta["models"] = []
        for i in range(len(clear_set.meta["models"])):
            x = clear_set.meta["models"][i]
            if x["name"] == self.name and x["suffix"] == suffix and x["config"] == self.config:
                os.remove(clear_set.meta["models"][i]["path"])
                del clear_set.meta["models"][i]
                break

        x = {"name": self.name, "suffix": suffix, "config": copy.deepcopy(self.config), "path": model_path}
        clear_set.meta["models"].append(x)
        clear_set.save_meta()
        self.log.debug("Saved model {} under {}".format(suffix, model_path))

    def deanonymize_all(self):
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        set = TorchImageDataset.from_set(self.dataset)

        for i in range(len(set)):
            x = set[i].to(device)
            z = self.model(x.view([-1] + self.model.input))
            set.update(i, z.view([3, 224, 224]))

    def cleanup(self):
        try:
            del self.model
            del self.optimizer
            del self.criterion
        except Exception:
            pass  # will fail if model was loaded not trained
        torch.cuda.empty_cache()
        self.log.info("Cleaned up.")
