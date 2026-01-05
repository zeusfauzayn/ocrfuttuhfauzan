import zipfile
import gzip
import os
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
import pickle

# Mapping EMNIST Balanced (0-46)
EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_data_from_zip():
    zip_path = os.path.expanduser('~/.cache/emnist/gzip.zip')
    extract_dir = 'temp_emnist'
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    print(f"Extracting relevant files from {zip_path}...")
    required_files = [
        'gzip/emnist-balanced-train-images-idx3-ubyte.gz',
        'gzip/emnist-balanced-train-labels-idx1-ubyte.gz',
        'gzip/emnist-balanced-test-images-idx3-ubyte.gz',
        'gzip/emnist-balanced-test-labels-idx1-ubyte.gz'
    ]
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in required_files:
            z.extract(file, extract_dir)
            
    # Read files
    base_dir = os.path.join(extract_dir, 'gzip')
    
    print("Loading training data...")
    x_train = read_idx(os.path.join(base_dir, 'emnist-balanced-train-images-idx3-ubyte.gz'))
    y_train = read_idx(os.path.join(base_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz'))
    
    print("Loading testing data...")
    x_test = read_idx(os.path.join(base_dir, 'emnist-balanced-test-images-idx3-ubyte.gz'))
    y_test = read_idx(os.path.join(base_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz'))
    
    return x_train, y_train, x_test, y_test

def train():
    try:
        x_train, y_train, x_test, y_test = load_data_from_zip()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # EMNIST images are rotated 90 degrees and flipped. Need to transpose.
    # Normalization
    print("Preprocessing data...")
    x_train = np.array([np.transpose(img) for img in x_train])
    x_test = np.array([np.transpose(img) for img in x_test])
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    num_classes = 47
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # Model Architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model (Balanced: Digits & Letters)...")
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

    print("Evaluating...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    if not os.path.exists('model'):
        os.makedirs('model')
        
    model.save('model/emnist_cnn.h5')
    with open('model/mapping.pkl', 'wb') as f:
        pickle.dump(EMNIST_MAPPING, f)
        
    print("Model saved to model/emnist_cnn.h5")

if __name__ == "__main__":
    train()
