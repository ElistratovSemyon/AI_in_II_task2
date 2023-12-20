import os
import numpy as np
import torch 
from dalle_model import SemanticCompressor
from utils import ImageTextDataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to model config")
    parser.add_argument("--data_path", type=str, help="path to directory with images and texts")
    parser.add_argument("--gpu", type=str, default=None, help="gpu num")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    dataset = ImageTextDataset(args.data_path)
    model = SemanticCompressor(args.config_path)
    model.run(dataset)

