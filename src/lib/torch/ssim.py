from pytorch_msssim import SSIM, MS_SSIM


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))


class MSSSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(MSSSIM_Loss, self).forward(img1, img2))
