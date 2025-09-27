import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split

# Cargar los datos
dfdata = pd.read_csv("diabetes.csv")

print(dfdata.dtypes)

# Preprocesar los datos
X = dfdata.drop(labels='Outcome', axis=1)
Y = dfdata.Outcome

# Dividir en entrenamiento y prueba
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Crear un modelo de red neuronal con Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(xtrain.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Predicción binaria (0 = No Diabetes, 1 = Diabetes)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

# Guardar el modelo en formato TensorFlow.js
tfjs.converters.save_keras_model(model, '\modelo_tensorflowjs')

