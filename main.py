from model import NST

from utils.args import get_args
from utils.image import ImageProcessor


def main():
    args = get_args()
    image_processor = ImageProcessor(args)
    style_img, content_img = image_processor.process()

    style_transfer = NST()
    style_transfer.run_inference(args, style_img, content_img, save=True)


if __name__ == "__main__":
    main()
