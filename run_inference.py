"""
Method adapted from GMIC function `run_model` which is licensed under a GNU Affero General Public License v3.0.
See: https://github.com/nyukat/Mammo-DETR/blob/main/LICENSE
"""

import os
import sys
import argparse
import pickle

from src.scripts.run_model import run_single_model

print(sys.version, sys.platform, sys.executable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mammo-DETR inference on the sample data')
    parser.add_argument('--model_path', default='models/')
    parser.add_argument('--data_path', default='sample_output/data.pkl')
    parser.add_argument('--image_path', default='sample_output/cropped_images')
    parser.add_argument('--segmentation_path', default='sample_data/segmentation')
    parser.add_argument('--output_path', default='sample_output')
    parser.add_argument('--device_type', default="gpu")
    parser.add_argument("--gpu_number", type=int, default=0)
    parser.add_argument("--model_index", type=str, choices=['1', '2', '3'], default="1")
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path
    }

    valid_model_index = ["1", "2", "3"]
    assert args.model_index in valid_model_index, "Invalid model_index {0}. Valid options: {1}".format(args.model_index, valid_model_index)
    # create directories
    os.makedirs(args.output_path, exist_ok=True)

    # set percent_t for the model
    single_model_path = os.path.join(args.model_path, "model_checkpoints_{0}.pt".format(args.model_index))
    output_dict = run_single_model(single_model_path, args.data_path, parameters)

    # save the predictions
    with open(os.path.join(args.output_path, "predictions.pkl"), 'wb') as f:
        pickle.dump(output_dict, f)
