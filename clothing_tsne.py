import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


encoded_outfits_file = "encoded_outfits_1678398851.9455404.pt"
dataset_dir = "D:\Workzone\AI Projects\Clustring Experiment\Dataset"




# Define the pan function
def on_key(event):
    axtemp = event.inaxes
    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()

    x_scale = (x_max - x_min) / 10.0
    y_scale = (y_max - y_min) / 10.0

    if event.key == 'w':
        # pan up
        axtemp.set(ylim=(y_min + y_scale, y_max + y_scale))
    elif event.key == 'a':
        # pan left
        axtemp.set(xlim=(x_min - x_scale, x_max - x_scale))
    elif event.key == 'x':
        # pan down
        axtemp.set(ylim=(y_min - y_scale, y_max - y_scale))
    elif event.key == 'd':
        # pan right
        axtemp.set(xlim=(x_min + x_scale, x_max + x_scale))
    fig.canvas.draw_idle()


# Define the zoom function
def on_scroll(event):
    axtemp = event.inaxes
    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()

    x_scale = (x_max - x_min) / 10.0
    y_scale = (y_max - y_min) / 10.0

    if event.button == 'up':
        # zoom in
        axtemp.set(xlim=(x_min + x_scale, x_max - x_scale))
        axtemp.set(ylim=(y_min + y_scale, y_max - y_scale))
    elif event.button == 'down':
        # zoom out
        axtemp.set(xlim=(x_min - x_scale, x_max + x_scale))
        axtemp.set(ylim=(y_min - y_scale, y_max + y_scale))
    fig.canvas.draw_idle()

def on_click(event):
    if points.contains(event)[0]:
        ind = points.contains(event)[1]["ind"][0]
        image_path = os.path.join(dataset_dir, clothe_names[ind])
        img = mpimg.imread(image_path)
        figg, ax = plt.subplots()
        ax.imshow(img)
        plt.show()


def main():
    global points
    global clothe_names
    global fig
    clothing_nodes = []

    encoded_clothes = torch.load(encoded_outfits_file)
    clothe_names = []
    for outfit in tqdm(encoded_clothes):
        encoded_tensor = encoded_clothes[outfit]
        encoding = encoded_tensor.to('cpu').numpy()
        clothing_nodes.append(encoding[0])
        clothe_names.append(outfit)

    clothing_nodes = np.array(clothing_nodes)
    clothing_dataframe = pd.DataFrame(clothing_nodes)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(clothing_dataframe)

    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    fig, ax = plt.subplots()
    points = ax.scatter(x, y)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)


    plt.show()


if __name__ == "__main__":
    main()
