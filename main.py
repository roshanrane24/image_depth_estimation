import os
import torch
import cv2
import utils
import argparse
from torchvision.transforms import Compose
from midas.mononet import MidasNet
from midas.transform import Resize, NormalizeImage, PrepareForNet

def run(input_path, output_path, model_path):
    """
    <<<
    >>>
    """

    print(f"Starting Operations")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    #Load Model 
    model = MidasNet(model_path)

    transform = Compose([Resize(
                                384,
                                384,
                                resize_target=None,
                                keep_aspect_ratio=True,
                                ensure_multiple_of=32,
                                resize_method="upper_bound",
                                image_interpolation_method=cv2.INTER_CUBIC
                        ),
                                            NormalizeImage(
                                       mean=[0.485, 0.456, 0.406], 
                                       std=[0.229,0.224,0.225]
                        ),
                        PrepareForNet()
                        ]
                )

    model.to(device)
    model.eval()
    
    # get input
    img_names = os.listdir(input_path)
    img_names = [image for image in img_names if image.endswith((".jpg", "jpeg"))]
    num_img = len(img_names)
    
    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("Start Processing Images")

    for idx, img_name in enumerate(img_names):
        print(f"[{idx+1}/{num_img}] Processing {img_name}")

        img = utils.read_image(os.path.join(input_path, img_name))
        img_input = transform({"image": img})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            prediction = model.forward(sample)
            prediction = (torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.shape[:2],
                            mode="bicubic",
                            align_corners=False).squeeze().cpu().numpy()
                        )

            filename = os.path.join(output_path, img_name.split('.')[0])
            utils.write_depth(filename, prediction)
        print(f"Finished processing {img_name}")
    print("Finished Processing")

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to input", default="./input")
    parser.add_argument("-o", "--output", help="path to output", default="./output")
    parser.add_argument("-m", "--model", help="path to model file", default="./model.pt")

    args = parser.parse_args()
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # compute depth maps
    run(args.input, args.output, args.model)
