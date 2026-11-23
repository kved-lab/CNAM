from sklearn.datasets import load_files
import numpy as np
from PIL import Image
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Chemin du dataset
dataset_path = "C:\\Users\\frede\\Repo\\Datalake\\archive\\original"


# Chargement des chemins et des labels
data = load_files(
    dataset_path,
    load_content=False,
    shuffle=True
)

# Récupération des chemins des images
image_paths = data.filenames

# Récupération des labels
labels = data.target

# Noms des classes
class_names = data.target_names

print("Nombre d'images :", len(image_paths))
print("Nombre de classes :", len(class_names))
print("Classes :", class_names)

class_counts = Counter(labels)

for i, class_name in enumerate(class_names):
    print(f"{class_name} : {class_counts[i]} images")

print("Dimension des images")


image_dimensions = []

for path in image_paths:
    try:
        img = Image.open(path)
        width, height = img.size
        channels = len(img.getbands())

        image_dimensions.append(
            (width, height, channels)
        )

    except Exception as e:
        print("Image non valide :", path)

df_dimensions = pd.DataFrame(
    image_dimensions,
    columns=["Largeur", "Hauteur", "Canaux"]
)

print(df_dimensions.describe())
print(
    df_dimensions.value_counts().head(10)
)


print("Distribution des classes")

class_distribution = pd.DataFrame({
    "Classe": class_names,
    "Nombre_images": [
        class_counts[i] for i in range(len(class_names))
    ]
})

class_distribution["Pourcentage"] = (
    class_distribution["Nombre_images"] /
    class_distribution["Nombre_images"].sum()
) * 100

print(class_distribution)


print ("Visualisation de la distribution")

import matplotlib.pyplot as plt
import numpy as np

# Calcul du nombre d'images par classe
class_counts = np.bincount(labels)

# Visualisation
plt.figure(figsize=(12, 6))

plt.bar(class_names, class_counts)

plt.title("Distribution des images par classe")
plt.xlabel("Classes")
plt.ylabel("Nombre d'images")

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


print("Visualisation de la distribution en pourentage")
plt.figure(figsize=(8, 8))

plt.pie(
    class_distribution["Nombre_images"],
    labels=class_distribution["Classe"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Répartition des images par classe")
plt.show()

print("Déterminer si le dataset est équilibré")
max_images = class_distribution["Nombre_images"].max()
min_images = class_distribution["Nombre_images"].min()

print("Classe majoritaire :",
      class_distribution.loc[
          class_distribution["Nombre_images"].idxmax(),
          "Classe"
      ])

print("Classe minoritaire :",
      class_distribution.loc[
          class_distribution["Nombre_images"].idxmin(),
          "Classe"
      ])

print("Ratio de déséquilibre :", round(max_images / min_images, 2))


print(" étude des idmensions des images")

from collections import Counter
from PIL import Image

dimensions = []

for path in image_paths:
    try:
        with Image.open(path) as img:
            dimensions.append(img.size)
    except Exception:
        continue

dimension_counts = Counter(dimensions)

print("Nombre de dimensions différentes :",
      len(dimension_counts))

print("\nDimensions les plus fréquentes :")

for dimension, count in dimension_counts.most_common(10):
    print(dimension, ":", count, "images")
