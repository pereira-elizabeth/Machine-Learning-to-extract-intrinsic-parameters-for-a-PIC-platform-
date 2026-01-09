# picml: ML utilities for parameter inference on PIC spectra
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from tensorflow.keras import layers, regularizers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========
#  Case A
# ==========

def create_model_A_1(input_shape):
    x_input = Input(shape=input_shape, name="x_input")
    x = Dense(40, activation="relu", kernel_initializer="glorot_uniform")(x_input)
    x = Dense(20, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = Dense(10, activation="relu", kernel_initializer="glorot_uniform")(x)
    output = Dense(1)(x)
    model = Model(x_input, output)
    return model

def create_model_A_2(input_shape):
    l2_reg = 5e-4  # Slightly lighter than before
    dropout_conv = 0.2
    dropout_dense = 0.4
    
    x_input = Input(shape=input_shape, name="x_input")
    
    # Small Gaussian noise at input for robustness
    x = layers.GaussianNoise(0.02)(x_input)
    
    # Conv Block 1
    x = layers.Conv1D(32, 7, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_conv)(x)
    
    # Conv Block 2
    x = layers.Conv1D(64, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_conv)(x)
    
    # Conv Block 3
    x = layers.Conv1D(64, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense Layers
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_dense)(x)
    
    output = layers.Dense(input_shape[0], activation='linear')(x)
    
    model = Model(inputs=x_input, outputs=output)
    return model


# ==========
#  Case B
# ==========

def create_model_B_1(input_shape):
    x_input = Input(shape=input_shape, name="x_input")
    x = Dense(40, activation="relu", kernel_initializer="glorot_uniform")(x_input)
    x = Dense(30, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = Dense(20, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = Dense(10, activation="relu", kernel_initializer="glorot_uniform")(x)
    x = Dense(5, activation="relu", kernel_initializer="glorot_uniform")(x)
    output = Dense(1)(x)
    model = Model(x_input, output)
    return model

def create_model_B_2(input_dim=40, l2=1e-4, p_drop=0.25, spatial_drop=0.15):
    # expects X shaped (None, input_dim, 1); y shaped (None, input_dim)
    x_in = Input(shape=(input_dim, 1), name="x_input")

    x = layers.Conv1D(32, 7, padding='same',
                      kernel_regularizer=regularizers.l2(l2))(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout1D(spatial_drop)(x)

    x = layers.Conv1D(48, 5, padding='same',
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # light downsample without adding params
    x = layers.MaxPooling1D(pool_size=2)(x)

    # global pooling keeps params tiny and resists overfit
    x = layers.GlobalAveragePooling1D()(x)

    # tiny head
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(p_drop)(x)

    y_out = layers.Dense(input_dim, activation='linear', name="y_out")(x)
    return Model(inputs=x_in, outputs=y_out)

def create_model_C_1(input_shape): #model to prevent overfitting
    # Input layer
    x_input = Input(shape=input_shape, name="x_input")
    # First Dense layer with L2 regularization and Dropout
    x = Dense(10*80, activation="relu", kernel_initializer="glorot_uniform", 
              kernel_regularizer=l2(0.0001))(x_input)
    x = Dropout(0.4)(x)  # Dropout with a rate of 0.3
    # Second Dense layer with L2 regularization and Dropout
    x = Dense(10*60, activation="relu", kernel_initializer="glorot_uniform", 
              kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)  # Dropout with a rate of 0.3
    # Third Dense layer with L2 regularization and Dropout
    x = Dense(10*40, activation="relu", kernel_initializer="glorot_uniform", 
              kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)  # Dropout with a rate of 0.3
    # Output layer
    output = Dense(2)(x)
    # Create and return the model
    model = Model(x_input, output)
    return model

  
# ==========
#  Case C
# ==========


def create_model_C_2(input_shape):
    l2_reg = 1e-3  # Adjust if needed

    x_input = Input(shape=input_shape, name="x_input")

    x = layers.Conv1D(64, 7, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    output = layers.Dense(input_shape[0], activation='linear')(x)

    return Model(inputs=x_input, outputs=output)
