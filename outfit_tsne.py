import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


encoded_outfits_file = "encoded_outfits_1678398851.9455404.pt"
outfits_txt = "outfits_1678368287.0405574_check.txt"
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

        outfit_names[0]

        image_paths = [os.path.join(dataset_dir, clothe_name) for clothe_name in outfit_names[ind]]      
        imgs = [mpimg.imread(img_path) for img_path in image_paths]

        figg, ax = plt.subplots(1,2)

        ax[0].imshow(imgs[0])
        ax[1].imshow(imgs[1])
        plt.show()






def readOutfitsTxt():
    outfits = []
    with open(outfits_txt, 'r') as f:
        for line in tqdm(f):
            outfits.append(line.strip().split(","))
    return outfits













def prepareCordinates():
    global outfit_names
    
    encoded_clothes = torch.load(encoded_outfits_file)

    outfist = readOutfitsTxt()

    outfit_nodes = []
    outfit_names = []

    for outfit in tqdm(outfist):
        try:
            encodings = []

            for clothe in outfit:
                    encoded_tensor = encoded_clothes[clothe]
                    encoding = encoded_tensor.to('cpu').numpy()[0]
                    encodings.append(encoding)

            outfit_encoding = np.concatenate((encodings[0], encodings[1]), axis=None)
            #outfit_encoding = np.mean(np.array(encodings), axis=0)


            outfit_nodes.append(outfit_encoding)
            outfit_names.append(outfit)

        except KeyError:
            pass

    outfit_nodes = np.array(outfit_nodes)
    outfit_dataframe = pd.DataFrame(outfit_nodes)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(outfit_dataframe)

    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    return x, y




def main():
    global points
    global fig
  
    x, y = prepareCordinates()

    fig, ax = plt.subplots()
    points = ax.scatter(x, y)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)


    plt.show()


if __name__ == "__main__":
    main()
