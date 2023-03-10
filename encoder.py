import torch
from tqdm import tqdm
import encoder_model as encoder
import numpy as np
import time
import os
from PIL import Image


outfits_txt = "outfits_1678368287.0405574_check.txt"
outfits_images_dir = "D:\Workzone\AI Projects\Clustring Experiment\Dataset"

def read_image(image):
    img = Image.open(os.path.join(outfits_images_dir, image))
    return img


def read_outfit_txt():
    outfits = []
    with open(outfits_txt, 'r') as f:
        for line in f:
            outfits.append(line.strip().split(","))
    return outfits

def flatten_2d_list(list_2d):
    return [item for sublist in list_2d for item in sublist]

def main():

    ## Initialize the model
    print("Initializing the model")
    encoder.initialize()

    print("Reading outfits")
    outfits = read_outfit_txt()
    outfits = flatten_2d_list(outfits)

    print("Encoding outfits")

    encoded_outfits = {}

    for outfit in tqdm(outfits):
        try:
            output = encoder.encode(read_image(outfit))
            encoded_outfits[outfit] = output
        except FileNotFoundError:
            print("File not found: {}".format(outfit))
            pass
    

    print("Saving encoded outfits")
    timestamp = str(time.time())
    torch.save(encoded_outfits, "encoded_outfits_{}.pt".format(timestamp))


if __name__ == "__main__":
    main()
