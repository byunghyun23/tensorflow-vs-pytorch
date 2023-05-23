import click
import numpy as np
import pandas as pd
import pickle
import os
import cv2
from tensorflow.python.keras.saving.save import load_model


def load_data(images_dir):
    name_list = []
    image_list = []

    files = os.listdir(images_dir)

    for file in files:
        try:
            path = images_dir + '/' + file

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (227, 227))

            name_list.append(file)
            image_list.append(image)

        except FileNotFoundError as e:
            print('ERROR : ', e)

    return np.array(name_list), np.stack(image_list)


@click.command()
@click.option('--data_dir', default='data/test', help='Data path')
@click.option('--model_name', default='model_tf', help='Model name')
def run(data_dir, model_name):
    image_names, X_test = load_data(data_dir)

    loaded_model = load_model(model_name + '.h5')

    pred = loaded_model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
        pred = [labels[k] for k in pred]

    results_df = pd.DataFrame({'image_names': image_names, 'class': pred})
    results_df.to_csv('results_tf.csv', index=False)


if __name__ == '__main__':
    run()