import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import tqdm
from MNISTS_dataloader import MnistDataloader
import random
import math
from pathlib import Path


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def train(model, dataloader, epochs, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}"
        )

        checkcpoint_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": running_loss / len(dataloader),
        }
        torch.save(checkcpoint_dict, f"checkpoint/checkpoint_epoch_{epoch + 1}.pth")


def plot_processed_images(model, dataset, num_img_show=25):

    num_shown = 0
    _ncols = 5
    _nrows = math.ceil(math.sqrt(num_img_show))
    random_dataset = [
        dataset[random.randint(0, len(dataset))] for _ in range(num_img_show)
    ]

    model.eval()
    with torch.no_grad():
        plt.figure(figsize=(10, 10))
        for image, label in tqdm.tqdm(random_dataset):
            if num_shown >= num_img_show:
                break
            output = model(image)
            _, predicted = torch.max(output, 1)

            plt.subplot(_nrows, _ncols, num_shown + 1)
            plt.imshow(image.squeeze(), cmap="gray")
            plt.title(f"Pred: {predicted.item()} Actual: {label}")
            plt.axis("off")
            num_shown += 1
        plt.show()


def main():
    # File paths for the MNIST dataset
    training_images_filepath = "archive/train-images-idx3-ubyte/train-images-idx3-ubyte"
    training_labels_filepath = "archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    test_images_filepath = "archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    test_labels_filepath = "archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

    # Initialize the MNIST data loader
    mnist_loader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )

    (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()

    train_dataset, test_dataset = MNISTDataset(x_train, y_train), MNISTDataset(
        x_test, y_test
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if Path("checkpoint_epoch_5.pth").is_file():
        checkpoint = torch.load("checkpoint_epoch_5.pth")
        model = SimpleNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model from checkpoint.")
    else:
        model = SimpleNN()
        train(model, train_dataloader, epochs=5, learning_rate=0.001)

    plot_processed_images(model, test_dataset, num_img_show=25)


if __name__ == "__main__":
    main()
