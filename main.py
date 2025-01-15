import argparse

from PIL import Image
import torch
import torchvision.transforms as T

from src.model import xU_NetFullSharp
from src.inference import remove_bone_shadow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    print(f"Using device: {device}")

    model = xU_NetFullSharp().to(device)

    print(f"Loading model weights from: {args.model_weights}")
    model.load_state_dict(torch.load(args.model_weights, weights_only=False, map_location=device))

    print(f"Loading input image: {args.input_image}")
    input_image = Image.open(args.input_image).convert('L')
    image_size = min(input_image.size[0], input_image.size[1])
    image_size = 2 ** (image_size.bit_length() - 1)
    if not args.no_limit_shape:
        image_size = min(image_size, 1024)
    input_image = input_image.resize((image_size, image_size), Image.BICUBIC)
    input_image = T.ToTensor()(input_image).to(device)

    print(f"Transformed input image size to: {input_image.shape}")

    print("Running inference...")
    remove_bone_shadow(model, input_image.unsqueeze(0), args.output_image, args.use_cmap if args.use_cmap else False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an image using the xU_NetFullSharp model.")

    parser.add_argument('--model_weights', '-w', type=str, required=True, help="Path to the model weights file.")
    parser.add_argument('--input_image', '-i', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_image', '-o', type=str, required=True, help="Path to save the output image.")
    parser.add_argument('--use_cmap', '-c', type=bool, help="Use a colormap for the output image.")
    parser.add_argument('--no_limit_shape', '-l', type=bool, help="Remove max image size limit of 1024px.")

    args = parser.parse_args()

    main(args)

    