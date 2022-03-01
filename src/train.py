import os
import sys
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EfficientNet
from .utils import set_random_seed

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

from config import DEVICE, MODEL_PATH


def run_training(
    train_dataloader,
    valid_dataloader,
    epochs,
    lr,
    random_seed=None,
    verbose=False,
    freeze_layers=True,
) -> float:

    if random_seed is not None:
        set_random_seed(random_seed=random_seed)

    model = EfficientNet(freeze=freeze_layers).model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    if freeze_layers:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()

    optimizer = torch.optim.SGD(model_parameters, lr=lr)

    min_epoch_validation_loss = np.inf
    for epoch in range(epochs):

        running_train_loss = 0.0
        running_train_images_count = 0

        model.train()
        for step, (images, labels) in enumerate(train_dataloader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            running_train_images_count += labels.size(0)

            if (step % 100 == 0) and verbose:

                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{step+1}/{len(train_dataloader)}], Running Train Loss: "
                    f"{round((running_train_loss/running_train_images_count), 4)}"
                )

        epoch_train_loss = running_train_loss / len(train_dataloader)

        # Evaluate Model
        print("Running model evaluation...")
        validation_accuracy, _, validation_loss = evaluate(
            model=model, dataloader=valid_dataloader
        )

        epoch_validation_loss = validation_loss / len(valid_dataloader)

        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {round(epoch_train_loss,4)}, "
            f"Valid Loss: {round(epoch_validation_loss, 4)}, "
            f"Valid Acc: {validation_accuracy} %"
        )

        if epoch_validation_loss < min_epoch_validation_loss:

            # Save Model
            print(
                f"Valdiation loss decreased from {min_epoch_validation_loss} to "
                f"{epoch_validation_loss}, saving the model..."
            )
            torch.save(model.state_dict(), MODEL_PATH)
            min_epoch_validation_loss = epoch_validation_loss

    return (validation_accuracy, epoch_validation_loss)


def evaluate(model, dataloader) -> typing.Union[float, torch.Tensor]:

    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():

        total_loss, batch_accuracy, total_images = 0.0, 0.0, 0
        batch_ground_labels, batch_probabilities = [], []
        for _, (batch_images, batch_labels) in enumerate(dataloader):

            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            batch_probabilities.append(probabilities)
            batch_ground_labels.append(batch_labels)
            total_images += batch_labels.size(0)

            batch_accuracy += (predicted == batch_labels).sum().item()

        return (
            (100 * batch_accuracy / total_images),
            batch_probabilities,
            total_loss,
        )
