import os
import logging
from tqdm import trange

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch

import torchvision.transforms as transforms

from .losses import ContentLoss, StyleLoss
from .normalization import Normalization

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


class NST:
    def __init__(self):
        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    def run_inference(self, args, style, content, save):
        """Return `g` for defined style (`s`) and content (`c`) given args."""

        logging.info("Building the style transfer model.")

        self.vgg = models.vgg19(weights=True).features.to(args.device).eval()

        model, style_losses, content_losses = self.get_style_model_and_losses(
            args, style, content
        )

        content.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = optim.LBFGS([content])

        logging.info("Optimizing `g`.")

        with trange(args.num_steps) as t:
            for step in t:

                def closure():
                    with torch.no_grad():
                        content.clamp_(0, 1)

                    optimizer.zero_grad()
                    model(content)
                    style_score = 0
                    content_score = 0

                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss

                    style_score *= args.style_weight
                    content_score *= args.content_weight

                    loss = style_score + content_score
                    loss.backward()

                    t.set_postfix(
                        style_loss=style_score.item(),
                        content_score=content_score.item(),
                    )

                    return style_score + content_score

                optimizer.step(closure)

        with torch.no_grad():
            content.clamp_(0, 1)

        if save:
            unloader = transforms.ToPILImage()
            content = content.cpu().clone()
            content = content.squeeze(0)

            output_image = unloader(content)
            if not os.path.exists(f"imgs/{args.style}"):
                os.makedirs(f"imgs/{args.style}")

            output_image.save(f"imgs/{args.style}/output-img.jpg")

        return content

    def get_style_model_and_losses(self, args, style, content):
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(args.device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(args.device)

        normalization = Normalization(normalization_mean, normalization_std).to(
            args.device
        )
        model = nn.Sequential(normalization)

        ###

        content_losses = []
        style_losses = []

        i = 0
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_{}".format(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu_{}".format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        return model, style_losses, content_losses
