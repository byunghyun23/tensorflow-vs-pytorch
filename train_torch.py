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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchsummary


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


class ImageDataGenerator(Dataset):
    def __init__(self, image_paths, labels):
        self.image_list = []
        self.label_list = []

        for i in range(len(image_paths)):
            image = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
            image = cv2.resize(image, (227, 227))
            label = labels[i]

            self.image_list.append(image)
            self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        images = torch.FloatTensor(self.image_list[index]).permute(2, 0, 1)
        labels = torch.LongTensor([self.label_list[index]])

        return images, labels


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
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


@click.command()
@click.option('--data_dir', default='data/train', help='Data path')
@click.option('--batch_size', default=128, help='Batch size')
@click.option('--epochs', default=20, help='Epochs')
@click.option('--model_name', default='model_torch', help='Model name')
def run(data_dir, batch_size, epochs, model_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')

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
    train_dataset = ImageDataGenerator(train_df['Filepath'].tolist(), train_df['Label'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = ImageDataGenerator(valid_df['Filepath'].tolist(), valid_df['Label'].tolist())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Define model
    model = MyModel(class_num=class_num).to(device)
    torchsummary.summary(model, input_size=(3, 227, 227))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss().to(device)

    # Train model
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    train_accs = []
    valid_accs = []
    avg_train_accs = []
    avg_valid_accs = []

    print('Start training..')
    for epoch in range(1, epochs + 1):

        # train
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            target = target.squeeze(dim=-1)

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            pred = torch.max(output, 1)[1]
            train_accs.append(((target == pred).sum() / len(target)).cpu())


        # validation
        model.eval()

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                target = target.squeeze(dim=-1)

                loss = loss_function(output, target)

                valid_losses.append(loss.item())

                pred = torch.max(output, 1)[1]
                valid_accs.append(((target == pred).sum() / len(target)).cpu())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_acc = np.average(train_accs)
        valid_acc = np.average(valid_accs)
        avg_train_accs.append(train_acc)
        avg_valid_accs.append(valid_acc)

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        epoch_len = len(str(epochs))

        print_msg = (f'Epoch {epoch:>{epoch_len}}/{epochs:>{epoch_len}} ' +
                     f'loss: {train_loss:.5f} ' +
                     f'accuracy: {train_acc:.5f} ' +
                     f'val_loss: {valid_loss:.5f} ' +
                     f'val_accuracy: {valid_acc:.5f}')

        print(print_msg)



    history = {'loss': avg_train_losses, 'val_loss': avg_valid_losses,
               'accuracy': avg_train_accs, 'val_accuracy': avg_valid_accs}

    pd.DataFrame(history)[['accuracy', 'val_accuracy']].plot()
    plt.title("Accuracy")
    plt.show()

    pd.DataFrame(history)[['loss', 'val_loss']].plot()
    plt.title("Loss")
    plt.show()

    torch.save(model.state_dict(), model_name + '.pt')

    # Map the label
    mapping = dict(zip(class_label.classes_, range(len(class_label.classes_))))
    labels = (mapping)
    labels = dict((v, k) for k, v in labels.items())
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    run()
