import torch
from torch.nn.functional import cosine_similarity
from bin.arcface.backbones import get_model
from torchvision.transforms import Resize


def cos2metric(x, y):
    return 100 * (1 - cosine_similarity(x, y).mean())


class ArcFace_Loss(torch.nn.Module):
    @torch.no_grad()
    def __init__(self):
        super().__init__()
        self.net = get_model("r50", fp16=False)
        self.net.load_state_dict(torch.load("arcface.pt"))
        self.net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.resize = Resize((112, 112))

    def forward(self, img1, img2):
        return cos2metric(self.net(self.transform(img1)), self.net(self.transform(img2)))

    def transform(self, img):
        return self.resize(img).sub_(0.5).div_(0.5)
