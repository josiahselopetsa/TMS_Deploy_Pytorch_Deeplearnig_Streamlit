# Import PyTorch

from __future__ import print_function, division
import torch

# We use torchvision to get our dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms
import streamlit as st
import cv2
# Display the first image
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
import torch.nn as nn
import os
import time
import copy
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torchvision import models



def imshow(img, tab):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    tab.pyplot(plt)


#______________________________________________________________________________________________

st.set_page_config(page_title="DeepLearningCV on Streamlit", page_icon="🎛️",layout="wide")
st.sidebar.markdown("### DeepLearningCV Part 2 🎛️")
#______________________________________________________________________________________________
# Streamlit UI components
st.title("Deep Learning CV with Streamlit Part 2")
tab2,tab3 = st.tabs([ "23. PyTorch Transfer Learning and Fine Tuning",""])  # ,"G","R","Hue"

# Check if a GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"GPU available: {torch.cuda.is_available()}")
st.sidebar.info(f"Using device: {device}")
# Images.image(image_rgb, caption="Processed Image", use_column_width=True, channels="RBG")

#______________________________________________________________________________________________
# Allow user to input the number of epochs
batch_size = st.sidebar.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
momentum = st.sidebar.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
lr = st.sidebar.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# "23. PyTorch Transfer Learning and Fine Tuning"
Learning_Tuning2  = tab2.selectbox(" PyTorch Transfer Learning and Fine Tuning",
                                   ["Create our data Loaders", "Fine Tuning the Convnet","ConvNet as fixed feature extractor"])
Learning_Tuning2_options = tab2.expander(" PyTorch Transfer Learning and Fine Tuning")
Learning_Tuning2_inputs = tab2.expander("Results : ")
placeholder2 = tab2.empty()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set to your image path
data_dir = './pages/hymenoptera_data'

# Use ImageFolder to point to your dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
               ['train', 'val']}

# Get dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


if Learning_Tuning2 == "Create our data Loaders":

    # Streamlit: Display dataset sizes and class names
    st.write(f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")
    st.write(f"Class names: {class_names}")


    # Helper function to show images
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Streamlit: Display a batch of images from the training set
    if st.button('Show Batch of Training Images'):
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # Plot using Matplotlib
        plt.figure(figsize=(10, 10))
        imshow(out, title=[class_names[x] for x in classes])

        # Streamlit: Display the plot
        st.pyplot(plt)

if Learning_Tuning2 == "Fine Tuning the Convnet":
    # Global variable for model_ft
    model_ft = None

    # Training model function adapted for Streamlit
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # Create a progress bar in Streamlit
        progress = st.progress(0)

        for epoch in range(num_epochs):
            st.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
            st.write('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                st.write('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            # Update progress bar
            progress.progress((epoch + 1) / num_epochs)

        time_elapsed = time.time() - since
        st.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        st.write('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    # Function to visualize predictions adapted for Streamlit
    def visualize_predictions(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(10, 10))

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        st.pyplot(fig)
                        return
            model.train(mode=was_training)



    # Define your model and training pipeline inside Streamlit
    def run_training():
        # Load the pretrained ResNet18 model
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features

        # Update the final fully connected layer for our dataset (2 classes in this example)
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=momentum)

        # Learning rate scheduler to decay LR by 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(model_ft,
                               criterion,
                               optimizer_ft,
                               exp_lr_scheduler,
                               num_epochs=epochs)

        # After training, display predictions
        st.write("Training complete. Visualizing predictions...")
        # visualize_predictions(model_ft)


    # Streamlit app layout
    st.title('Image Classification with ResNet')

    # Button to run training
    if st.button('Start Training'):
        run_training()

    # Optionally, you can provide a way to visualize only without retraining
    if st.button('Visualize Predictions'):
        if model_ft is not None:
            visualize_predictions(model_ft)
        else:
            st.write("Model not trained yet. Please train the model first.")

if Learning_Tuning2 == "ConvNet as fixed feature extractor":
    pass

# ----------------------------------------------------------------------------------------------------------------------

