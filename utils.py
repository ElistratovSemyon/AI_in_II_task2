import numpy as np
from PIL import Image

from functools import partial, wraps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Pad, Resize, ToPILImage, InterpolationMode, Normalize
from dalle2_laion.scripts import InferenceScript
import os

def pad_transform(img, w, h):
    return Pad(padding=(0, 0, w, h))(img)
    

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.img_dir = os.path.join(data_path, "images")
        self.text_dir = os.path.join(data_path, "texts")
        self.img_filenames = sorted(os.listdir(self.img_dir))
        self.text_filenames = sorted(os.listdir(self.text_dir))
        self.pad_tr = pad_transform
        self.resize_tr = Resize((256, 256), 
                                interpolation=InterpolationMode.BILINEAR)
        
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_filenames[idx]))
        w, h = img.size
        img = ToTensor()(img)
        new_size = max(w, h)
        img = self.pad_tr(img, new_size - w, new_size - h)
        img = self.resize_tr(img)

        with open(os.path.join(self.text_dir, self.text_filenames[idx]), 'r') as text_file:
            text = [text.strip() for text in text_file.readlines()]

        return img, text

    def __len__(self):
        return len(self.img_filenames)

        
class ExampleInference(InferenceScript):
    def run(self, text: str):
        """
        Takes a string and returns a single image.
        """
        text = [text]
        image_embedding_map = self._sample_prior(text)
        image_embedding = image_embedding_map[0][0].unsqueeze(0)
        image_map = self._sample_decoder(text=text, image_embed=image_embedding)
        return image_map[0][0]
