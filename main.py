#Esta libreria funciona para poder conectarse con la url de manera segura
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Esta libreria lo que hace es ayudar a poder importar un dataset de CIFAR-10 de ejemplo completo con el 100% de los datos
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
