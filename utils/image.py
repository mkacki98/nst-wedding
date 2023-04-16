import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image


class ImageProcessor:
    def __init__(self, args):
        self.device = args.device
        self.style = args.style

        self.imsize = 512
        self.loader = transforms.Compose(
            [transforms.Resize(self.imsize), transforms.ToTensor()]
        )

        self.unloader = transforms.ToPILImage()

    def process(self):
        style_img = Image.open(f"imgs/styles/{self.style}.jpg")
        content_img = Image.open("imgs/content-img.jpg")

        style_img, content_img = self.resize_images(style_img, content_img)

        style_img = self.image_loader(style_img)
        content_img = self.image_loader(content_img)

        return style_img, content_img

    def image_loader(self, image):
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def resize_images(self, style_img, content_img):
        style_resized = style_img.resize(content_img.size)
        return style_resized, content_img
