import os
from PIL import Image

# Chemins des datasets
source_dir = "C:\\Users\\frede\\Repo\\Datalake\\archive\\original"
destination_dir = "C:\\Users\\frede\\Repo\\Datalake\\archive\\original_resized"

# Dimensions souhaitées
target_size = (224, 224)

# Extensions des images acceptées
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Compteur
total_images = 0
errors = 0

# Parcours des dossiers et sous-dossiers
for root, dirs, files in os.walk(source_dir):

    # Reproduire l'arborescence
    relative_path = os.path.relpath(root, source_dir)
    output_folder = os.path.join(destination_dir, relative_path)

    os.makedirs(output_folder, exist_ok=True)

    for file in files:

        if file.lower().endswith(extensions):

            image_path = os.path.join(root, file)

            output_path = os.path.join(output_folder, file)

            try:
                # Ouvrir l'image
                with Image.open(image_path) as img:

                    # Conversion en RGB
                    img = img.convert("RGB")

                    # Redimensionnement
                    img = img.resize(
                        target_size,
                        Image.Resampling.LANCZOS
                    )

                    # Enregistrement
                    img.save(output_path, quality=95)

                    total_images += 1

            except Exception as e:
                errors += 1
                print(f"Erreur : {image_path} : {e}")

print("Redimensionnement terminé !")
print("Nombre d'images traitées :", total_images)
print("Nombre d'erreurs :", errors)
print("Répertoire de destination :", destination_dir)