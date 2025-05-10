import cv2
import numpy as np
from torchvision import transforms

tile_size = 512

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((tile_size, tile_size)),
    transforms.ToTensor(),
])

def extract_tiles(image, tile_size=512):
    h, w, _ = image.shape
    tiles = []
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
    return tiles
