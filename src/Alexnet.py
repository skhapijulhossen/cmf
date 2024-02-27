from tensorflow.keras.layers import Input, Dense, Conv2D, Add
from tensorflow.keras.layers import SeparableConv2D, ReLU
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization, Dense, Embedding, Layer, Dropout, Input
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten, Activation, MaxPooling2D, add, GlobalMaxPooling2D



def buildAlexNet(inputs):
    y = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
               activation='relu', input_shape=(227, 227, 3))(inputs)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(y)
    y = Conv2D(filters=256, kernel_size=(5, 5), strides=(
        1, 1), activation='relu', padding="same")(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(y)
    y = Conv2D(filters=384, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', padding="same")(y)
    y = BatchNormalization()(y)
    y = Conv2D(filters=384, kernel_size=(1, 1), strides=(
        1, 1), activation='relu', padding="same")(y)
    y = BatchNormalization()(y)
    y = Conv2D(filters=256, kernel_size=(1, 1), strides=(
        1, 1), activation='relu', padding="same")(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(y)
    y = Flatten()(y)
    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)
    return y
