# ReconocimientoFacturas
Red Neuronal para reconocimiento/lectura de facturas, remitos y tickets.

### Armar Base de Datos (csv):

```javascript
imagen,ruta_archivo,texto_asociado
imagen1.jpg,/ruta/imagen1.jpg,Texto de la imagen 1
imagen2.jpg,/ruta/imagen2.jpg,Texto de la imagen 2
imagen3.jpg,/ruta/imagen3.jpg,Texto de la imagen 3
...
```
Faltaria definir las columnas, como "producto", "precio unitario", "cantidad" u otras.

### Arquitectura 
- CNN para el procesamiento de las imagenes
- NLP para el procesamiento de lenguaje natural

### Bosquejo Implementación
```python
import tensorflow as tf
from tensorflow import keras

# Definir la arquitectura del modelo
def create_model():
    # Modelo para procesamiento de imágenes
    image_model = keras.Sequential([
        # Capas convolucionales, de pooling, etc.
        # Añade las capas que necesites para procesar las imágenes
    ])

    # Modelo para procesamiento de texto
    text_model = keras.Sequential([
        # Capas de procesamiento de texto, como la tokenización y la representación de palabras
        # Añade las capas que necesites para procesar el texto
    ])

    # Combinar las salidas de los modelos de imagen y texto
    combined_model = keras.layers.concatenate([image_model.output, text_model.output])
    # Añadir capas adicionales para fusionar las salidas

    # Capa de salida
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(combined_model)

    # Crear el modelo final
    model = keras.models.Model(inputs=[image_model.input, text_model.input], outputs=output_layer)

    return model

# Crear el modelo
model = create_model()

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento
model.fit([train_images, train_text], train_labels, epochs=10, batch_size=32)

# Evaluar el modelo con los datos de prueba
loss, accuracy = model.evaluate([test_images, test_text], test_labels)

# Hacer predicciones con el modelo
predictions = model.predict([input_image, input_text])
```

#### Bosquejo Modelo CNN
```python
def create_image_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    return model
```

#### Bosquejo Modelo NLP
```python
def create_text_model(vocab_size, embedding_dim, max_length):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dense(64, activation='relu'))
    return model
```
