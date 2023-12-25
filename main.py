import os
import torch
import pandas as pd
from PIL import Image
from shutil import copy
from lavis.models import load_model_and_preprocess

from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser


def process_image(
    id: int,
    image_path: str,
    model: dict,
    vis_processors: dict,
    cls_names: list
) -> str:
    """
        This function will get the class with highest properties, then copy 
    Args:
        id (int): The id of the pictures (To create new name)
        image_path (str): The path of the image
        model (dict): The model used to process the image
        vis_processors (dict): The processors used to process the image
        cls_names (list): The list of the class names which will be used to classify the image

    Return: 
        The new name of the image without the extension
    """
    try:
        image_name = os.path.basename(image_path)
        extention = os.path.splitext(image_name)[1]

        # Load the image
        raw_image = Image.open(image_path).convert("RGB")
        # Resize the image
        raw_image = raw_image.resize((596, 437))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        sample = {"image": image, "text_input": cls_names}

        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
        text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]

        sims = (image_features @ text_features.t())[0] / model.temp

        probs = torch.nn.Softmax(dim=0)(sims).tolist()

        # Get the class with highest properties
        max_prob = max(probs)
        max_prob_index = probs.index(max_prob)

        # Get the class name
        class_name = cls_names[max_prob_index]

        if class_name == 'motorcycle':
            if max_prob < 0.7:
                print(f"Threshold is not reached 0.7 for image {image_path}")
                class_name = 'other'

    except Exception as e:
        print(f"Error when processing image {image_path}: {e}")
        return None
    else:
        return {
            'is_motorcycle': class_name == 'motorcycle',
            'old': image_name,
            'new': f"{id}_{class_name}{extention}"
        }


def main(input_folder: str, output_folder: str):
    # Get the list of the images
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess(
        "blip_feature_extractor", model_type="base", is_eval=True, device=device)

    class_names = ["motorcycle", "car", "road", "sky"]

    # Create the mapping
    mapping = {}
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs('not_motorcycle', exist_ok=True)

    for id, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        result = process_image(id, image_path, model, vis_processors, class_names)

        if result is not None:
            if result['is_motorcycle']:
                old_name = result['old']
                new_name = result['new']
                mapping[old_name] = new_name

                print(f"[{id+1:>6} | {len(image_files):<6}] ===> Renaming {old_name} to {new_name}")

                copy(os.path.join(input_folder, old_name), os.path.join(output_folder, new_name))
            else:
                old_name = result['old']
                new_name = result['new']
                mapping[old_name] = new_name

                print(f"[{id+1:>6} | {len(image_files):<6}] ===> Renaming {old_name} to {new_name}")

                copy(os.path.join(input_folder, old_name), os.path.join('not_motorcycle', new_name))

    # Save the mapping
    df = pd.DataFrame.from_dict(mapping, orient='index', columns=['new_name'])
    df.index.name = 'old_name'
    df.to_csv('mapping.csv')


if __name__ == "__main__":
    # Get the arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str, default="data", help="The folder which contains the images"
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="output", help="The folder which contains the output images with labels images"
    )

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
