#Esta libreria funciona para poder conectarse con la url de manera segura
import ssl

import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

#Esta libreria lo que hace es ayudar a poder importar un dataset de CIFAR-10 de ejemplo completo con el 100% de los datos
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential  # Importamos Sequential para construir el modelo
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization  # Capas necesarias para la CNN

# Deshabilitamos la verificación de SSL para evitar errores de certificado
ssl._create_default_https_context = ssl._create_unverified_context

# Cargamos el dataset CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

def probarDataset():
    """Función que imprime las dimensiones de los datos de entrenamiento y pruebas
    y muestra una imagen de ejemplo del conjunto de entrenamiento."""
    print("Dimensiones de X_train:", X_train.shape)
    print("Dimensiones de y_train:", y_train.shape)
    print("Dimensiones de X_test:", X_test.shape)
    print("Dimensiones de y_test:", y_test.shape)

    # Mostrar una imagen del dataset
    plt.imshow(X_train[0])
    plt.title(f"Etiqueta: {y_train[0][0]}")
    plt.show()

def definimosClases():
    """Función que muestra una cuadrícula de imágenes aleatorias del conjunto de entrenamiento
    etiquetadas con sus respectivas clases."""
    # Definimos las etiquetas de las clases de CIFAR-10
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Configuramos una cuadrícula de 10x10 para mostrar imágenes aleatorias
    W_grid, L_grid = 10, 10
    fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
    axes = axes.ravel()  # Aplanamos el array de ejes

    # Recorremos la cuadrícula para mostrar imágenes aleatorias
    n_train = len(X_train)
    for i in np.arange(0, W_grid * L_grid):
        index = np.random.randint(0, n_train)  # Selección aleatoria de índice
        axes[i].imshow(X_train[index])
        label_index = int(y_train[index][0])  # Convertimos el índice de etiqueta a entero
        axes[i].set_title(labels[label_index], fontsize=8)
        axes[i].axis('off')

    # Ajustar los espacios entre imágenes
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def distribucionClasesEntrenamiento():
    """Función que muestra la distribución de clases en el conjunto de entrenamiento mediante un gráfico de barras."""
    classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    classes, counts = np.unique(y_train, return_counts=True)
    plt.barh(classes_name, counts)
    plt.title('Distribución de clases en el conjunto de entrenamiento')
    plt.xlabel('Cantidad de Imágenes')
    plt.show()

def distribucionClasesTest():
    """Función que muestra la distribución de clases en el conjunto de prueba mediante un gráfico de barras."""
    classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    classes, counts = np.unique(y_test, return_counts=True)
    plt.barh(classes_name, counts)
    plt.title('Distribución de clases en el conjunto de pruebas')
    plt.xlabel('Cantidad de Imágenes')
    plt.show()

def preprocesadoDatos():
    """Función que normaliza los valores de píxeles en los conjuntos de datos de entrenamiento y prueba
    escalándolos de [0, 255] a [0, 1] para facilitar el entrenamiento de modelos de deep learning."""
    global X_train, X_test  # Declaramos las variables como globales para modificarlas
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    #one-hot-encoding vectors
    """Esto significa que cada etiqueta de clase se representará como un vector binario de longitud 
    igual al número de clases (10 en CIFAR-10), lo cual es ideal para redes neuronales."""
    y_categorical_train = to_categorical(y_train, num_classes = 10)
    y_categorical_test = to_categorical(y_test, num_classes = 10)

    # Verificación de las dimensiones de salida
    print("Dimensiones de y_categorical_train:", y_categorical_train.shape)
    print("Dimensiones de y_categorical_test:", y_categorical_test.shape)
    print("--------------------------------")
    print(y_categorical_train)
    print(y_categorical_test)
    print("--------------------------------")
    print(y_train)

def crearRedNeuronal():
    #Ulizamos el metodo sequential que  el standar para las CNN
    #Averiguamos que tipo de red debemos utilizar(cambia siempre dependiendo del proyecto)
    #Construimos la red por medio de capas y lo iniciamos con model = Sequential()
    model = Sequential()

    #=========>Convulutional layer<================
    #Añadimos la primera capa (Conv2D añade una capa convolucional)
    model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    #La normalizacion por Lotes es un metodo que se utiliza para hacer que las redes neuronales artificiales sean
    #mas rapidas y estables mediante la normalizacion de las entradas de las capas al volver a centrar y escalar
    model.add(BatchNormalization())

def deep_learning():
    """Función principal que ejecuta los distintos métodos para explorar y visualizar el dataset."""
    #probarDataset()
    #definimosClases()
    #distribucionClasesEntrenamiento()
    #distribucionClasesTest()
    preprocesadoDatos()
    crearRedNeuronal()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    deep_learning()