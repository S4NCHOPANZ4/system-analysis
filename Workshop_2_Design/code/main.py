import os
import glob
import torch
import torch.nn.functional as F
import cv2
import pandas as pd
from model import ResNet50_CBAM
from utils import extract_tiles, transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify_folder(folder_path, model):
    model.eval()
    abs_path = os.path.abspath(folder_path)
    print(f"🔎 Buscando imágenes en: {abs_path}")

    tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))
    if not tiff_files:
        print("⚠️ No se encontraron archivos .tif.")
        return

    results = []

    for file_path in tiff_files:
        print(f"\n📂 Procesando imagen: {os.path.basename(file_path)}")
        image = cv2.imread(file_path)
        if image is None:
            print(f"❌ Error al cargar imagen: {file_path}")
            continue

        tiles = extract_tiles(image)
        if not tiles:
            print(f"⚠️ No se extrajeron tiles de la imagen: {file_path}")
            continue

        for idx, tile in enumerate(tiles):
            input_tensor = transform(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                output = F.softmax(model(input_tensor), dim=1)
            pred = output.cpu().numpy()[0]

            # Guardamos el resultado en una fila de la tabla
            results.append({
                "Imagen": os.path.basename(file_path),
                "Tile": idx,
                "Glomerulus": round(pred[0], 4),
                "Blood Vessel": round(pred[1], 4),
                "Unsure": round(pred[2], 4)
            })

    # Convertir a DataFrame y mostrar como tabla
    if results:
        df = pd.DataFrame(results)
        print("\n📊 Resultados en tabla:")
        print(df.to_string(index=False))
    else:
        print("⚠️ No se generaron resultados para mostrar.")

if __name__ == "__main__":
    print("🚀 Inicializando modelo ResNet50_CBAM...")
    model = ResNet50_CBAM(num_classes=3).to(device)
    classify_folder("../../Workshop-I/data/train", model)
