# Import library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import click
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def generate_data_frame(data_dir):
    dir = Path(data_dir)
    filepaths = list(dir.glob(r'**/*.jpg'))

    labels = [str(filepaths[i]).split("\\")[-2] \
              for i in range(len(filepaths))]

    filepath = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    df = pd.concat([filepath, labels], axis=1)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    return df


def image_data_generator(image_paths, labels):
    image_list = []
    label_list = []

    for i in range(len(image_paths)):
        image = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (227, 227))
        label = labels[i]

        image_list.append(image)
        label_list.append(label)

    return np.stack(image_list), np.array(label_list)


def my_model(class_num):
    input_shape = (227, 227, 3)
    x = tf.keras.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, activation='relu', strides=4)(x)
    pool1 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(conv1)  # overlapped pooling

    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu', strides=1, padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(conv2)

    conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', strides=1, padding='same')(pool2)
    conv4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', strides=1, padding='same')(conv3)
    conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same')(conv4)
    pool3 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(conv5)

    # FC
    f = tf.keras.layers.Flatten()(pool3)
    f = tf.keras.layers.Dropout(0.5)(f)
    f = tf.keras.layers.Dense(4096, activation='relu')(f)
    f = tf.keras.layers.Dropout(0.5)(f)
    f = tf.keras.layers.Dense(4096, activation='relu')(f)
    out = tf.keras.layers.Dense(class_num, activation='softmax')(f)

    model = tf.keras.Model(inputs=x, outputs=out)

    return model


@click.command()
@click.option('--data_dir', default='data/train', help='Data path')
@click.option('--batch_size', default=128, help='Batch size')
@click.option('--epochs', default=20, help='Epochs')
@click.option('--model_name', default='model_tf', help='Model name')
def run(data_dir, batch_size, epochs, model_name):
    # Load data
    df = generate_data_frame(data_dir)
    print('========== Data shape ==========')
    print(df.shape)
    print('========== Data ================')
    print(df)
    print()

    # Label encoding
    class_label = LabelEncoder()
    df['Label'] = class_label.fit_transform(df['Label'].values)
    print('========== Class ================')
    print(np.sort(df['Label'].unique()))
    print('========== Class number =========')
    class_num = len(df['Label'].unique())
    print(class_num)

    # Separate dataset
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=0)
    X_train, y_train = image_data_generator(train_df['Filepath'].tolist(), train_df['Label'].tolist())
    X_valid, y_valid = image_data_generator(valid_df['Filepath'].tolist(), valid_df['Label'].tolist())

    # Define model
    model = my_model(class_num=class_num)
    model.summary()
    optimizer = Adam(lr=1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs)

    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    plt.title("Accuracy")
    plt.show()

    pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
    plt.title("Loss")
    plt.show()

    model.save(model_name + '.h5')

    # Map the label
    mapping = dict(zip(class_label.classes_, range(len(class_label.classes_))))
    labels = (mapping)
    labels = dict((v, k) for k, v in labels.items())
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    run()
