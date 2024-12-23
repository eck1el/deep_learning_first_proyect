#Esta libreria funciona para poder conectarse con la url de manera segura
import ssl
import time

import numpy as np
import tensorflow as tf


ssl._create_default_https_context = ssl._create_unverified_context

#Esta libreria lo que hace es ayudar a poder importar un dataset de CIFAR-10 de ejemplo completo con el 100% de los datos
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential  # Importamos Sequential para construir el modelo
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout  # Capas necesarias para la CNN

from tensorflow.keras.callbacks import EarlyStopping

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

    crearRedNeuronal()
    entrenar(X_test, y_categorical_test, X_train, y_categorical_train)


def crearRedNeuronal():
    #Inicializamos el modelo como un modelo secuencial CNN
    #Averiguamos que tipo de red debemos utilizar(cambia siempre dependiendo del proyecto)
    #Construimos la red por medio de capas y lo iniciamos con model = Sequential()
    model = Sequential()

    # =======> Primera capa convolucional <=======
    # Añadimos una capa Conv2D con 32 filtros y un kernel (ventana) de tamaño 3x3.
    # input_shape define el tamaño de las imágenes de entrada: (32x32 píxeles con 3 canales RGB).
    # Usamos 'relu' como función de activación para añadir no linealidad.
    # 'same' asegura que la salida tenga el mismo tamaño que la entrada.
    model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    #La normalizacion por Lotes es un metodo que se utiliza para hacer que las redes neuronales artificiales sean
    #mas rapidas y estables mediante la normalizacion de las entradas de las capas al volver a centrar y escalar

    # Añadimos Batch Normalization para normalizar las salidas de la capa Conv2D.
    # Esto acelera el entrenamiento y mejora la estabilidad de la red.
    model.add(BatchNormalization())

    # =======> Segunda capa convolucional <=======
    # Otra capa Conv2D con los mismos parámetros para aumentar la capacidad de extracción de características.
    model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # =======> Capa de max-pooling <=======
    # Añadimos MaxPooling2D para reducir la dimensionalidad de las características extraídas.
    # pool_size=(2, 2) toma la ventana más grande de 2x2, reduciendo el tamaño de la entrada a la mitad.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Dropout reduce el overfitting(Sobreajuste)
    # =======> Dropout para reducir el sobreajuste <=======
    # Dropout "apaga" aleatoriamente un 25% de las neuronas para evitar que el modelo se sobreajuste a los datos de entrenamiento.
    model.add(Dropout(0.25))

    # =======> Tercera capa convolucional <=======
    # Incrementamos los filtros a 64 para capturar características más complejas.
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # =======> Cuarta capa convolucional <=======
    # Otra capa Conv2D con los mismos parámetros que la anterior.
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # =======> Segunda capa de max-pooling <=======
    # Reducimos aún más las dimensiones con otra capa MaxPooling2D.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =======> Dropout <=======
    # Aplicamos nuevamente Dropout, apagando el 25% de las neuronas para reducir el sobreajuste.
    model.add(Dropout(0.25))

    # =======> Quinta capa convolucional <=======
    # Aumentamos los filtros a 128, ya que estamos aprendiendo características aún más complejas.
    model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # =======> Sexta capa convolucional <=======
    # Otra capa Conv2D con 128 filtros y los mismos parámetros.
    model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # =======> Tercera capa de max-pooling <=======
    # Reducimos las dimensiones nuevamente con MaxPooling2D.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =======> Dropout <=======
    # Aplicamos Dropout con 25% para reducir el sobreajuste en esta etapa final de extracción de características.
    model.add(Dropout(0.25))

    #Añadimos capa Flatten. Utilizada para hacer que la entrada multidimensional sea unidimensinal,
    #Comunmente utilizada en la transicion de la capa de convolucion a la capa final
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    #Añadimos una capa final softmax para que podamos clasificar las imagenes
    #Tenemos que indicar el numero de clases que tiene el problema
    #Este trabajador toma las decisiones(decide entre las 10 categorias cual es el resultado)
    model.add(Dense(10, activation='softmax'))

    #Compilamos el modelo
    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    #Este es como el entrenador del equipo
    #Este entrenador se va a asegurar que todas las capas entrenen de la mejor manera posible
    #patience = 2 significa 2 epocas y esto lo que hace es que verifica 2 veces si el entrenamiento debe detenerse
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)

    #Decide cuantas veces debe de entrenar para dar un mejor resultado
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    model.summary()

def entrenar(X_test, y_categorical_test, X_train, y_categorical_train):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    #Aqui creamos mas imagenes para entrenamiento basado en las imagenes que ya tenemos pero rotandolas para que se van un poco diferente
    batch_size = 32
    data_generator = ImageDataGenerator(width_shift_range = 0.1, height_shift_range=0.1, horizontal_flip=True)
    train_generator = data_generator.flow(X_train, y_categorical_train, batch_size)
    steps_per_epoch = X_train.shape[0] // batch_size

    start_time = time.time()
    r = model.fit(train_generator,
                  epochs = 50,
                  steps_per_epoch = steps_per_epoch,
                  validation_data = (X_test, y_categorical_test),
                  )
    print("--- Tiempo: %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
    historico(r)
    guardar_modelo_deep_learning_opcion1()
    guardar_modelo_deep_learning_opcion2()
    evaluamos_modelo(r)
    mostramos_porcentaje_precision(X_test, y_categorical_test)
    creamos_matriz_confusion(X_test, y_test)


def historico(r):
    #la variable r nos permite crear un historico
    loss_hist = r.history['val_loss']
    acc_hist = r.history['val_accuracy']
    print(f"val_loss->{len(loss_hist)} -> {min(loss_hist)} -> {max(loss_hist)}")
    print(f"acc_hist ->{len(acc_hist)} -> {min(acc_hist)} -> {max(acc_hist)}")


def guardar_modelo_deep_learning_opcion1():
    from tensorflow.keras.models import load_model
    model.save('cnn_50_epochs.h5')

def guardar_modelo_deep_learning_opcion2():
    import h5py
    with h5py.File('cnn_50_epoch_v2.h5', 'w') as f:
        model.save(f)


def evaluamos_modelo(r):
    #Mostraremos 4 graficos de la evolucion de las funciones
    #loss, precision, accuracy y recall
    plt.figure(figsize=(12, 16))

    #La perdida
    #Es una de las metricas principales para evaluar un modelo
    plt.subplot(4, 2, 1)
    plt.plot(r.history['loss'], label='Loss')
    plt.plot(r.history['val_loss'], label="val_Loss")
    plt.title('Loss Function Evolution')
    plt.legend()

    #Precision
    #Nos mide la calidad de la prediccion
    plt.subplot(4, 2, 3)
    plt.plot(r.history['precision'], label=['accuracy'])
    plt.plot(r.history['val_precision'], label='val_precision')
    plt.title('Precision Function Evolution')
    plt.legend()

    #Recall
    #Nos indica la cantidad de veces que esta acertando
    plt.subplot(4, 2, 4)
    plt.plot(r.history['recall'], label=['recall'])
    plt.plot(r.history['val_recall'], label='val_recall')
    plt.title('Recall Function Evolution')
    plt.legend()

    #Accuracy
    #Es solo para corroborar los datos de los otros graficos
    #Entre mas nos acerquemos a 1 es mejor
    plt.subplot(4, 2, 2)
    plt.plot(r.history['accuracy'], label='accuracy')
    plt.plot(r.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy Function Evolution')
    plt.legend()

def mostramos_porcentaje_precision(X_test, y_categorical_test):
    #Mostramos el % de precision basado en el accuracy
    evaluation = model.evakuate(X_test, y_categorical_test)
    print(f'Test Accuracy : {evaluation[1]*100:.2f}%')

def creamos_matriz_confusion(X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import confusionMatrixDisplay

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')

    plt.show()

    classification_report(X_test, y_test, y_pred)


def classification_report(X_test, y_test, y_pred):
    #classification report
    #obtenemos los valores de precision, recall, f1-score y support para cada una de las categorias de prediccion(perros, gatos, etc)
    print(classification_report(y_test, y_pred))

    probamos_imagen(X_test, y_test)
    probamos_dataset_completo(y_pred)

def probamos_imagen(X_test, y_test):
    # Aqui lo que estamos haciendo es una validacion individual con una sola imagen
    # cogemos la imagen 101 del conjunto de test solo como ejemplo

    my_image = X_test[101]
    plt.imshow(my_image)

    #nos devuelve el numero de la categoria del animal que se predice
    y_test[101]

    #predecimos el animal nuevamente mostrando el numero de categoria de animal que es
    np.argmax(model.predict(my_image.reshape(1, 32, 32, 3)))


def probamos_dataset_completo(y_pred):
    #Tambien tenemos esta forma para probar todo el dataset
    #Definimos las etiquetas o clases de nuestro problema
    #Nombre común: Labels
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Configuramos una cuadrícula de 10x10 para mostrar imágenes aleatorias
    W_grid, L_grid = 10, 10
    fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
    axes = axes.ravel()  # Aplanamos el array de ejes

    #obtenemos el tamaño del dataset de pruebas
    n_test = len(X_test)

    #Recorremos la matriz de imagenes
    for i in np.arange(0, W_grid * L_grid):
        #Seleccionamos un numero aleatorio entre 0 y n_test
        index = np.random.randint(0, n_test)

        #Leer y mostrar la imagen con el indice seleccionado
        axes[i].imshow(X_test[index,1:])
        label_index = int(y_pred[index])
        axes[i].set_title(labels[label_index], fontsize = 8)
        axes[i].axis('off')

    #Ajustar los espacios entre las iamgenes de la matriz resultante
    plt.subplots_adjust(hspace=0.4)

    #Obtenemos las predicciones
    predictions = model.predict(X_test)

    imagen_con_grafica(predictions)


def imagen_con_grafica(predictions):
    def plot_image(i, predictions_array, true_label, img, labels):
        # Aqui generamos una visualizacion de las predicciones con cada imagen y un grafico a la par
        # Si la prediccion es correcta me lo muestra en azul y con que categoria, y si fue erronea(rojo)
        # Esto tambien me va a indicar con cual categoria se equivoco
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel(
            f"{labels[int(predicted_label)]} {100 * np.max(predictions_array):2.0f}% ({labels[int(true_label)]})",
            color=color)

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, int(true_label[i])
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    num_rows = 8
    num_cols = 5
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows)
        plot_image(i, predictions[i], y_test, X_test)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], y_test)
    plt.tight_layout()
    plt.show()


def probar_modelo_con_imagenes_nuevas():
    from skimage.transform import resize
    #Aqui es realmente donde probamos si el modelo funciona
    my_image = plt.imread("cats_1000/cat.0.jpg")
    my_image_resized = resize(my_image, 32, 32, 3)
    img = plt.imshow(my_image_resized)
    probabilities = model.predict(np.array([my_image_resized,]))
    print(probabilities)

    number_to_class =

def deep_learning():
    """Función principal que ejecuta los distintos métodos para explorar y visualizar el dataset."""
    #probarDataset()
    #definimosClases()
    #distribucionClasesEntrenamiento()
    #distribucionClasesTest()
    preprocesadoDatos()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    deep_learning()