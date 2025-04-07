
import cv2, numpy as np
import pandas as pd

#lista para input data
X_data = []

#lista para labels
y_labels = []

yo = r"\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data\resized_new_pfsense1.png_resized.jpg"

no_yo = r"\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data\resized_new_pfsense2.png_resized.jpg"

# LA IMAGEN VERDADERA TENDRA LABEL 1
image_yo = cv2.imread(yo)
X_data.append(image_yo)
y_labels.append(1)

# LA IMAGEN FAKE TENDRA LABEL 0
image_no_yo = cv2.imread(no_yo)
X_data.append(image_no_yo)
y_labels.append(0)

#HACER X_DATA LISTA DE NP
X_sets = np.array(X_data)
# CANTIDAD DE MUESTRAS
X_sets.shape[0]

#APLANAR LAS IMAGENES PARA QUE SEAN UN VECTORES DE UNA DIMENSION
# Y SU TRANSPONER PARA QUE SEA UN VECTOR (FEATURE, NUMERO DE IMAGENES)
feature_array = X_sets.reshape(X_sets.shape[0], -1).T

#LABELS UBICADAS EN 2D, (1, N Imagenes)
y_labels = np.array(y_labels)
y_labels = y_labels.reshape(1, -1)

#NORMALIZAR LOS VALORES DE LOS PIXELES DE 0 A 1
x = feature_array/255.0



print(x)
print(y_labels)