import numpy as np
import cv2
import os
#PARAMETROS DE IMAGEN
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
"""
Función para cargar imágenes y convertirlas en vectores normalizados.
    @param image_paths: Lista de rutas de imágenes.
    @param labels: Lista de etiquetas correspondientes a las imágenes.
    
    @return: Dos arrays numpy, uno con las imágenes vectorizadas y otro con las etiquetas.
"""

def vectorize_images(image_paths, labels):
    
    X_data = []
    y_data = []

    # Combinar en tuplas, (image_path, label) ex: ("img1.jpg", 1)
    for path, label in zip(image_paths, labels):
        #cargar imagen y convertirla a escala de grises para evaluar intensidad más que color    
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Error cargando imagen en: {path}")
        # redimensionar
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # establecer los datos
        X_data.append(image)
        y_data.append(label)
    
    # Aplana las imágenes para que sean vectores de una dimensión
    # y transponer para que sea un vector (feature, número de imágenes)
    X_array = np.array(X_data).reshape(len(X_data), -1).T 
    #labels ubicadas en 2D, (1, N imágenes)
    y_array = np.array(y_data).reshape(1, -1)
    
    #Normalizar los valores de los píxeles para que esten entre 0 a 1
    X_array = X_array / 255.0
    return X_array, y_array


class NN:
    #Definición del constructor
    def __init__(self, input_size, hidden_size, output_size):
        """Valores de entrada, capas ocultas y salida de forma paramétrica 

        Args:
            input_size (int): tamaño de la imagen (600*800)
            hidden_size (int): número de neuronas en la capa oculta
            output_size (int): número de neuronas en la capa de salida
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        """
        Inicializacion de pesos y sesgos
        Aleaotorios con distribución normal (media=0, varianza=0.01)
        W1(peso), b1(sesgo) capa de Entrada -> Capa Oculta
        """
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        
        """
        W2(peso), b2(sesgo) capa de Oculta -> Capa Salida
        """
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    """
    Func de Activación Sigmoide
    Cualquier valor real lo convierte en un valor entre 0 y 1
        args:
            z(float): valor de entrada
        
        @return: valor de salida entre 0 y 1
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    """
    Derivada de la función sigmoide
    Sirve para actualizar los pesos con gradientes cuando se hace backpropagation
        args:
            z(float): valor de la función sigmoide
        
        @return: derivada de la función sigmoide
    """
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    """
    Propagación hacia adelante, datos entran, atraves de la red y se obtiene una predicción
        args:
            X (array): datos de entrada (imágenes vectorizadas)
        
        @return: salida de la red (predicción)
    """ 
    def forward(self, X):
        # z1 = (W.X) + b1
        self.z1 = np.dot(self.W1, X) + self.b1
        # a1 = sigmoide(z1), Capa de entrada -> capa oculta
        self.a1 = self.sigmoid(self.z1)
        # z2 = (W.a1) + b2
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        # a2 = sigmoide(z2), Capa oculta -> capa de salida
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    """
    Ajusta los pesos y sesgos usando el error entre predicción y etiqueta real
        args:
            X (array): datos de entrada (imágenes vectorizadas)
            y (array): etiquetas reales
            learning_rate (float): tasa de aprendizaje para la actualización de pesos
        
    """
    def backward(self, X, y, learning_rate=0.01):
        # m samples
        m = X.shape[1]
        # dz2 = a2 - y, diferencia en la predicción y la etiqueta real
        dz2 = self.a2 - y
        # dW2 = (1/m)(dz2.a1^T), gradiente de W2
        dW2 = (1 / m) * np.dot(dz2, self.a1.T)
        # db2 = (1/m)(sum(dz2)), gradiente de b2
        # keepdims=True mantiene la misma dimensión que b2
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        # dz1 = (W2^T.dz2) * sigmoide'(a1), propagar error hacia la capa oculta de atrás
        dz1 = np.dot(self.W2.T, dz2) * self.sigmoid_derivative(self.a1)
        # dw1 = (1/m)(dz1.X^T), gradiente de W1
        dW1 = (1 / m) * np.dot(dz1, X.T)
        # db1 = (1/m)(sum(dz1)), gradiente de b1
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        
        # Actualizar los pesos y sesgos usando la tasa de aprendizaje descenso de gradiente
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    """
    proceso de entrenamiento de la red neuronal
    args:
        X_train (array): datos de entrenamiento (imágenes vectorizadas)
        y_train (array): etiquetas de entrenamiento
        epochs (int): número de épocas para el entrenamiento
        learning_rate (float): tasa de aprendizaje para la actualización de pesos
    """
    def train(self, X_train, y_train, epochs=800, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X_train)
            self.backward(X_train, y_train, learning_rate)
            if epoch % 50 == 0:
                # Calcular la pérdida (error cuadrático medio)
                # loss = (1/m) * sum((a2 - y)^2)
                loss = np.mean((self.a2 - y_train) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    """
    predicción de la red neuronal
    args:
        X (array): datos de entrada (imágenes vectorizadas)

    @return: salida de la red (predicción)
    La salida es un valor entre 0 y 1, donde 0 indica que no es la persona y 1 indica que sí lo es.
    """
    def predict(self, X):
        return self.forward(X)

