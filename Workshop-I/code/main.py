import cv2
import pickle
import json, math
import numpy as np
import random, time
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import shutil, sys, os, gc
import matplotlib.pyplot as plt


# to remove a file or a folder in kaggle directory
def delete_file_or_directory(path):
    """
    Delete a file or directory if it exists.

    Parameters:
        path (str): Path to the file or directory.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print(f"{path} has been deleted.")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"{path} and its contents have been deleted.")
    else:
        print(f"{path} does not exist.")
        
delete_file_or_directory("./HuMap")

# To read 'tif' images
def image_from_tif(df):
    image_fpath = "../data/train/"
    images = [{'id': row['id'], 'img': cv2.imread(image_fpath + row['id'] + ".tif")[:,:,::-1]} for _, row in tqdm(df.iterrows(), total=df.shape[0])]
    
    return np.array(images)

# To read 'polygons.json' and return an image
def mask_from_json(data, image_shape=(512, 512, 3)):
    image = np.zeros(image_shape, dtype=np.uint8)
    
    for annotation in data['annotations']:
        coordinates = np.array(annotation['coordinates'])
        
        if annotation['type'] == 'blood_vessel':
            color = (255, 0, 0)  # red for glomerulus
            cv2.fillPoly(image, [coordinates], color=color)
        if annotation['type'] == 'glomerulus':
            color = (0, 255, 0)  # Green for blood vessel
            cv2.fillPoly(image, [coordinates], color=color)
        if annotation['type'] == 'unsure':
            color = (0, 0, 255)  # blue for unsure
            cv2.fillPoly(image, [coordinates], color=color)
    
    return image

# To read 'polygons.json' and return an image
def masks_info_df(annots):
    infos = []
    
    for annot in annots:
        info = {}
        info['id'] = annot['id']
        info['blood_vessel'] = 0
        info['glomerulus'] = 0
        info['unsure'] = 0
    
        for annotation in annot['annotations']:
            info[annotation['type']] = info[annotation['type']] + 1
            
        infos.append(info)
    
    return pd.DataFrame(infos)

tile_fpath = "../data/tile_meta.csv"
wsi_fpath = "../data/wsi_meta.csv"

tile_df = pd.read_csv(tile_fpath)
tile_df.head()
wsi_df = pd.read_csv(wsi_fpath)
wsi_df.head()

# Joining tile_df, and wsi_df on 'source_wsi'
kidney_df = pd.merge(tile_df, wsi_df, on='source_wsi', how='left')
kidney_df['dataset_wsi'] = kidney_df['dataset'].astype(str) + kidney_df['source_wsi'].astype(str)
print(kidney_df.shape)
kidney_df.head()

# Initialize an empty list to store the parsed JSON objects
annots = []

# Open the JSONL file
with open('../data/polygons.jsonl', 'r') as jsonl_file:
    # Iterate over each line in the file
    for line in jsonl_file:
        # Parse each line as JSON and append it to the list
        annots.append(json.loads(line))
masks = []

for annot in tqdm(annots, total=len(annots)):
    mask = {}
    mask['id'] = annot['id']
    mask['img'] = mask_from_json(annot)
    masks.append(mask)
    
masks = np.array(masks)

# Directory to save the masks
directory = "./HuPMap/masks/"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Iterate over the array of objects
for obj in masks:
    # Extract ID and image array
    id_ = obj['id']
    img_array = obj['img']
    
    # Save the image array as a numpy file with the ID as the filename
    np.save(f"{directory}{id_}.npy", img_array)

# intialising our masks df
mask_df = masks_info_df(annots)
mask_df['annotated'] = 1
print(mask_df.shape)
mask_df.head()

# Joining tile_df, and wsi_df on 'source_wsi'
kidney_df = pd.merge(kidney_df, mask_df, on='id', how='left')
kidney_df['annotated'] = kidney_df['annotated'].fillna(0)
print(kidney_df.shape)
kidney_df.head()

# ensuring everything is fine
print(kidney_df[kidney_df['annotated'] == 1].shape)
print(kidney_df[(kidney_df['annotated'] == 1) & (kidney_df['blood_vessel'] != 0)].shape)

# saving dataframe
kidney_df.to_csv('./HuPMap/kidney_tiles.csv', index=False)

df = kidney_df[kidney_df['annotated'] == 1]
print(df.shape)
df.head()

# reading images
images = image_from_tif(df)

# Directory to save the masks
directory = "./HuPMap/images/"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Iterate over the array of objects
for obj in images:
    # Extract ID and image array
    id_ = obj['id']
    img_array = obj['img']
    
    # Save the image array as a numpy file with the ID as the filename
    np.save(f"{directory}{id_}.npy", img_array)

tiles_fpath = "./HuPMap/kidney_tiles.csv"
kidney_df = pd.read_csv(tiles_fpath)
print(kidney_df.shape)
kidney_df.head()

# getting our annotated data only
kidney_df = kidney_df[kidney_df['annotated'] == 1]
print(kidney_df.shape)
kidney_df.head()

""" channel param when one is specified the mask will contain only 'blood_vessels' type,
when 2 is specified 'unsure' type will be added to 'blood_vessels' type,
any other option the mask will be returned with the 3 channels
"""
def read_mask(path, channel=2):
    # loading mask
    mask = np.load(path)
    
    # Select the specified channels
    if channel == 1:
        mask = mask[:, :, 0]
    elif channel == 2:
        # Sum the pixel values of selected channels along the last axis (channel axis)
        selected_channels = mask[:, :, [0, 2]]
        mask = np.sum(selected_channels, axis=2)
    else:
        pass
    
    # expanding dimension if needed
    if len(mask.shape) != 3:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        
    return mask

def load_kidney_tiles(df, read_mask=read_mask):
    fpath = "./HuPMap"
    X = np.array([np.load(f"{fpath}/images/{row['id']}.npy") for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="loading images")])
    y = np.array([read_mask(f"{fpath}/masks/{row['id']}.npy") for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="loading masks")])
    
    return X, y

idx = 522
x, y = load_kidney_tiles(kidney_df.iloc[idx:idx+1])
print(kidney_df.iloc[idx])

plt.imshow(x[0])
plt.imshow(y[0])
print(y[0].shape)