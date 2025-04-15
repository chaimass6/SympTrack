import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
# ✅ Sample Data
data = {
    'Fever': [1, 1, 0, 1],
    'Cough': [1, 0, 0, 1],
    'Fatigue': [1, 1, 1, 1],
    'Headache': [0, 1, 1, 1],
    'Breathlessness': [0, 0, 1, 0],
    'Disease': [0, 1, 2, 3]  # 0: Flu, 1: Typhoid, 2: Asthma, 3: Dengue
}

df = pd.DataFrame(data)

X = df.drop('Disease', axis=1).values
y = df['Disease'].values

# ✅ Build the Model
model = Sequential([
    Dense(16, input_shape=(5,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='softmax')  # 4 disease classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the Model
model.fit(X, y, epochs=100, verbose=0)

# ✅ Save as TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('swasthya_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model saved as swasthya_model.tflite")

# ✅ Load the Model
disease_labels = {
    0: "Flu",
    1: "Typhoid",
    2: "Asthma",
    3: "Dengue"
}

import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="swasthya_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Example: User Inputs (Fever, Cough, Fatigue, Headache, Breathlessness)
user_input = np.array([[1, 1, 1, 0, 0]], dtype=np.float32)  # Example for Flu

interpreter.set_tensor(input_details[0]['index'], user_input)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# ✅ Output the Prediction
disease_labels = {0: "dengue", 1: "Typhoid", 2: "Asthma", 3: "flu", 4:"fever" }
print("Predicted Disease:", disease_labels[predicted_class])
