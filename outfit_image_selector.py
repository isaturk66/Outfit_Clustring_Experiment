import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
import time
import concurrent.futures


# Script hyperparameters
path_to_product_pairs = "outfits_products_50000_check1.txt"
path_to_merged_image_urls = "C:\imageUrls.txt"
path_to_dataset_dir = "D:\Workzone\AI Projects\Clustring Experiment\Dataset"


# Define the batch size, input size, and number of classes
batch_size = 1
input_size = (224, 224)
num_classes = 2

# Define a transform to preprocess the data
transform = transforms.Compose([
    
    transforms.Resize(input_size),
                        
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

path_to_model= "./model_70_2023_01_08_02_43_59.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
# Define the model
def get_model():
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft
    

def calculate_probability(image):
    model.eval()
    with torch.no_grad():
        img = transform(image).to(device)

        # Forward pass
        outputs = model(img.unsqueeze(0))

        # Apply softmax to the outputs
        probs = F.softmax(outputs, dim=1)

        # Get the predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update the correct and total counts
        return (predicted[0].item(),  max( [prob.item() for prob in probs[0]]))

def check_file_exists(path):
    return os.path.exists(path)

def read_pairs_txt(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        return [line.strip().split(",") for line in lines]
    
def read_image_urls(path):
    lines = []

    with open(path, 'r') as f:
        for line in tqdm(f):
            lines.append(line.rstrip('\n'))

    return lines
            

def populate_product_to_image_dict(product_pairs):
    image_dict = {}
    #Flatten product pairs
    products = []
    for pair in product_pairs:
        products.append(pair[0])
        products.append(pair[1])

    #Read image urls
    print("Reading image urls")
    image_urls = read_image_urls(path_to_merged_image_urls)
    

    print("Populating product to image dict")
    for line in tqdm(image_urls):
        try:
            name, varient , url = line.strip().split(",")            

            if name.split("_")[0] not in image_dict:
                image_dict[name.split("_")[0]] = {}
            
            image_dict[name.split("_")[0]][name] = url
        
        
        except ValueError as e:
            pass

    
    filtered_image_dict = {}

    for product in products:
        filtered_image_dict[product] = image_dict[product]

    del product
    del image_dict

    return filtered_image_dict
    

def get_best_image(product):
    images = []
    names = []
    for varient in product:
        response = requests.get(product[varient])
        img = Image.open(BytesIO(response.content))
        images.append(img)
        names.append(varient)

    
    best_prob = 0
    best_image = None
    best_image_name = None


    for i in range(len(images)):
        try:
            image = images[i]
            name = names[i]
            
            prediction, probability = calculate_probability(image)
        

            prob = -1 * (prediction *2 - 1) * probability

            if prob > best_prob:
                best_prob = prob
                best_image = image
                best_image_name = name
        except RuntimeError as e:
            pass

    return best_image_name, best_image


def create_outfits(args):
    file, pair, product_to_image_dict = args
    images = []
    names = []
    
    for product in pair:
        name, best_image = get_best_image(product_to_image_dict[product])
        
        if best_image is None:
            return False

        images.append(best_image)
        names.append(name)

    
    
    for i in range(len(images)):
        if not check_file_exists(os.path.join(path_to_dataset_dir, names[i])):
            images[i].save(os.path.join(path_to_dataset_dir, names[i]))

    
    data_entry = ",".join(names)

    del images
    del names

    file.write(data_entry + '\n')
    file.flush()



def main():
    global model

    #Get Model
    model = get_model().to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(path_to_model))


    print("Reading product pairs")
    product_pairs =  read_pairs_txt(path_to_product_pairs)


    print("Populating product to image dict")
    product_to_image_dict = populate_product_to_image_dict(product_pairs)


    print("Creating outfits")
    timestamp = str(time.time())

    
    with open(f"outfits_{timestamp}.txt", 'a') as file:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for pair in product_pairs:
                future = executor.submit(create_outfits, [file, pair, product_to_image_dict])
                futures.append(future)
            
            # Monitor progress using tqdm
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
            
    print("Done")
    

if __name__ == "__main__":
    main()