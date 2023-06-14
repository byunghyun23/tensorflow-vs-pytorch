import click
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import torch
import torch.nn as nn


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

    print('torch.FloatTensor(image_list).shape', torch.FloatTensor(image_list).shape)
    images = torch.FloatTensor(image_list).permute(0, 3, 1, 2)
    names = np.array(name_list)

    return names, images


class MyModel(nn.Module):
    def __init__(self, class_num):
        super(MyModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, class_num)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

@click.command()
@click.option('--data_dir', default='data/test', help='Data path')
@click.option('--model_name', default='pytorch_model', help='Model name')
@click.option('--class_num', default=8, help='class number')
def run(data_dir, model_name, class_num):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')

    image_names, X_test = load_data(data_dir)

    loaded_model = MyModel(class_num).to(device)
    loaded_model.load_state_dict(torch.load(model_name + '.pt'))
    loaded_model.eval()

    with torch.no_grad():
        data = X_test.to(device)
        pred = loaded_model(data)
        pred = np.argmax(pred.cpu(), axis=1).detach().numpy()

    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
        pred = [labels[k] for k in pred]

    results_df = pd.DataFrame({'image_names': image_names, 'class': pred})
    results_df.to_csv('pytorch_results.csv', index=False)


if __name__ == '__main__':
    run()
