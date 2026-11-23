from sklearn.datasets import load_files

dataset_path = "C:\\Users\\frede\\Repo\\Datalake\\archive\\original_resized"

data = load_files(
    dataset_path,
    load_content=False,
    shuffle=False
)

image_paths = data.filenames
labels = data.target
class_names = data.target_names

print("Nombre d'images :", len(image_paths))
print("Nombre de classes :", len(class_names))
print("Classes :", class_names)

# . Transformer les images en vecteurs numériques

import numpy as np
from PIL import Image

X = []

for path in image_paths:

    try:
        with Image.open(path) as img:

            #img = img.convert("RGB")

            #img = img.resize(
            #    (32, 32),
            #    Image.Resampling.LANCZOS
            #)

            # Transformation en vecteur
            img_array = np.array(img).flatten()

            X.append(img_array)

    except Exception as e:
        print("Erreur :", path, e)

X = np.array(X, dtype=np.uint8)

print("Dimensions de X :", X.shape)
print("Dimensions de y :", labels.shape)

X = []
y = []

for path, label in zip(image_paths, labels):

    try:
        with Image.open(path) as img:
            #img = img.convert("RGB")
            #img = img.resize((32, 32))

            X.append(np.array(img).flatten())
            y.append(label)

    except Exception as e:
        print("Image ignorée :", path)

X = np.array(X, dtype=np.uint8)
y = np.array(y)


#. Diviser le dataset en apprentissage et test


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

print("y_train :", y_train.shape)
print("y_test :", y_test.shape)

#. Construire et entraîner l'arbre de décision


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Entraînement
model.fit(X_train, y_train)

print("Entraînement terminé !")


#Évaluer les performances du modèle

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Prédictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

# Rapport de classification
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

print("Matrice de confusion :")
print(cm)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest")

plt.title("Matrice de confusion - Decision Tree")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")

#Visualiser la matrice de confusion

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.colorbar()

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.tight_layout()
plt.show()


## Visaulaiser l'arbre de décision 

from sklearn.tree import plot_tree

plt.figure(figsize=(25, 12))

plot_tree(
    model,
    feature_names=None,
    class_names=class_names,
    filled=True,
    max_depth=3,
    fontsize=8
)

plt.title("Arbre de décision")
plt.show()