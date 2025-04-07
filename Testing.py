from nn import NN, vectorize_images

#PARAMETROS DE IMAGEN
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

# rutas
positive_images = [
    r"\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data\resized\resized_new_trainPic_resized.jpg",
]

negative_images = [
    r"\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data\resized\resized_new_trainMadonna_resized.jpg",
]

test_image_impostor = r"\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data\resized\resized_new_testPrince_resized.jpg"
test_image_true = r"\Users\marti\Documents\Carrera jaja\12_DUODECIMO_TRIMESTRE_2025\IA\project\data\resized\resized_new_testPic_resized.jpg"

image_paths = positive_images + negative_images
labels = [1] * len(positive_images) + [0] * len(negative_images)

# Carga de datos
X_train, y_train = vectorize_images(
    image_paths, labels
)

nn = NN(input_size=IMAGE_WIDTH * IMAGE_HEIGHT, hidden_size=30, output_size=1)
nn.train(X_train, y_train, epochs=300, learning_rate=0.01)

for test_path in [test_image_true, test_image_impostor]:
    X_test, _ = vectorize_images([test_path], [1])
    pred = nn.predict(X_test)[0][0]
    label = "MISMA PERSONA" if pred > 0.50 else "PERSONA DISTINTA"
    print(f"{test_path} → {label} ({pred*100:.2f}% confidence)")