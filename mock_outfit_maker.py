import random
import os
import time

outfit_sample_size = 100

dataset_root_path = "D:\Workzone\Datasets\Trendyol"

woman_upper_folders = ["kadin-bluz","kadin-ceket", "kadin-gomlek", "kadin-kazak", "kadin-mont", "kadin-tshirt", "kadin-kot-ceket",]
woman_lower_folders = ["kadin-etek", "kadin-pantalon", "kadin-jean"]

men_upper_folders = ["erkek-forma", "erkek-gomlek",  "erkek-kazak", "erkek-mont", "erkek-tshirt","erkek-yelek", "erkek-sweat", "erkek-ceket"]
men_lower_folders = ["erkek-esofman", "erkek-pantalon", "erkek-jean", "erkek-sort"]

men_upper_pool = []
men_lower_pool = []
woman_upper_pool = []
woman_lower_pool = []


def readTextFile(path):
    print("Reading file: " + path)
    with open(path, "r") as f:
        return [line.split(",")[0] for line in f.read().splitlines()]
    
def write_outfits(outfits):
    print("Writing outfits to file")
    timestamp = str(time.time())
    with open(f"outfits_products_{timestamp}.txt", "w") as f:
        for outfit in outfits:
            f.write(f"{outfit[0]},{outfit[1]}")
            f.write("\n")


def sampleProductLines(sample_size, path):
    lines = readTextFile(path)
    return random.sample(lines, sample_size)

def sampleFolderFamily(folder_family):
    print("Sampling folder family: " + str(folder_family))
    s_size = round((outfit_sample_size/2) / len(folder_family))
    family_lines = []

    for folder in folder_family:
        path = os.path.join(dataset_root_path, folder, "productUrls.txt")
        family_lines.extend(sampleProductLines(s_size, path))

    return family_lines

def create_pairs(upper_body_clothes, lower_body_clothes):
    print("Creating pairs")
    random.shuffle(upper_body_clothes)
    pairs = []
    for i in range(len(upper_body_clothes)):
        pairs.append((upper_body_clothes[i], lower_body_clothes[i]))
    return pairs



def main():
    woman_upper_pool.extend(sampleFolderFamily(woman_upper_folders))
    woman_lower_pool.extend(sampleFolderFamily(woman_lower_folders))
    men_upper_pool.extend(sampleFolderFamily(men_upper_folders))
    men_lower_pool.extend(sampleFolderFamily(men_lower_folders))

    woman_outfits = create_pairs(woman_upper_pool, woman_lower_pool)
    men_outfits = create_pairs(men_upper_pool,men_lower_pool)
    
    outfits = woman_outfits
    outfits.extend(men_outfits)

    write_outfits(outfits)
    

if __name__ == '__main__':
    main()
