import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definir um modelo simples para testar
model = Sequential([
    Dense(10, input_shape=(10,), activation='relu'),
    Dense(1)
])

save_dir = './save'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_save_path = os.path.join(save_dir, 'test_model.h5')
model.save(model_save_path)
print(f'Modelo salvo em: {model_save_path}')