
import torch
import torchvision
import torchvision
import torchvision.transforms as transforms
import streamlit as st
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix, classification_report

# import torchvision.transforms as transforms

from PIL import Image


def imshow(img, tab):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    tab.pyplot(plt)
#
#
# #______________________________________________________________________________________________
#
st.set_page_config(page_title="DeepLearningCV on Streamlit", page_icon="👢",layout="wide")
st.sidebar.markdown("### DeepLearningCV Part 1 👢")
#______________________________________________________________________________________________
# Streamlit UI components
st.title("Deep Learning CV with Streamlit Part 1")
tab1,tab4 = st.tabs(["1. PyTorch CNN Tutorial MNIST", "7. PyTorch - Fashion-MNSIT Part 1 - No Regularisation"])  # ,"G","R","Hue"


# Check if a GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"GPU available: {torch.cuda.is_available()}")
st.sidebar.info(f"Using device: {device}")
# Images.image(image_rgb, caption="Processed Image", use_column_width=True, channels="RBG")

#______________________________________________________________________________________________
MNIST = tab1.selectbox("PyTorch CNN Tutorial MNIST", ["Fetch our MNIST Dataset using torchvision", "Create our Data Loader", "Now we build our Model", "Training Our Model", "Reload the model we just saved"])
MNIST_options = tab1.expander("Fetch our MNIST Dataset using torchvision")
MNIST_inputs = tab1.expander("Input values: ")
placeholder1 = tab1.empty()

# Allow user to input the number of epochs

batch_size = st.sidebar.number_input("Number of batch_size:", min_value=1, max_value=1000, value=128, step=1)
epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100, value=2, step=1)
momentum = st.sidebar.number_input("Number of momentum:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
lr = st.sidebar.number_input("Number of lr:", min_value=0.000, max_value=0.010, value=0.001, step=0.0001)



# Transform to PyTorch tensors and normalize our values between -1 and +1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load our Training Data and specify the transform to use when loading
trainset = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)

# Load our Test Data and specify the transform to use when loading
testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

# Prepare train and test loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


# Define the neural network model using a class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 32, 3)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3)  # Second convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 10)  # Fully connected layer 2

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create an instance of the network and move it to the selected device
net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

if MNIST == "Fetch our MNIST Dataset using torchvision":

    # Display dataset shapes
    MNIST_options.write(f"Trainset shape: {trainset.data.shape}")
    MNIST_options.write(f"Testset shape: {testset.data.shape}")

    # This is the first value in our dataset
    MNIST_options.write(f"First training sample shape: {trainset.data[0].shape}")
    MNIST_options.write(trainset.data[0])


    # Define a function to display images using Streamlit
    def imgshow(title="", image=None, size=6):
        aspect_ratio = image.shape[0] / image.shape[1]
        MNIST_options.image(image, caption=title)


    # Convert image to a numpy array
    image = trainset.data[0].numpy()
    imgshow("MNIST Sample", image)

    # Display the first 50 MNIST images in tabs
    tab1.write("First 20 MNIST images:")

    # Create tabs for the first 50 images
    tabs = tab1.tabs([f"Image {i + 1}" for i in range(20)])

    # Display each image in its respective tab
    for index, tab in enumerate(tabs):
        with tab:
            tab.image(trainset.data[index].numpy(), width=28, caption=f"Label: {trainset.targets[index]}")

if MNIST == "Create our Data Loader":
    # We use the Python function iter to return an iterator for our trainloader object
    dataiter = iter(trainloader)

    # We use next to get the first batch of data from our iterator
    images, labels = next(dataiter)

    # Display the shapes of the images and labels
    tab1.write(f"Images shape: {images.shape}")
    tab1.write(f"Labels shape: {labels.shape}")


    # Function to show an image
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        tab1.image(np.transpose(npimg, (1, 2, 0)), caption="Batch of images", use_column_width=True)


    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images using Streamlit
    tab1.write("Batch of training images:")
    imshow(torchvision.utils.make_grid(images))

    # Print labels
    tab1.write(' '.join(f'{labels[j].item()}' for j in range(128)))

if MNIST == "Now we build our Model":


    # Display the model architecture in Streamlit
    MNIST_options.write("Model architecture:")
    MNIST_options.text(str(net))

    # Define the loss function (Cross Entropy Loss)
    criterion = nn.CrossEntropyLoss()

    # Display the loss function and optimizer in Streamlit
    tab1.write("Loss function: Cross Entropy Loss")
    tab1.write("Optimizer: SGD with learning rate = 0.001 and momentum = 0.9")

    # Display the optimizer details
    tab1.write("Optimizer details:")
    tab1.text(str(optimizer))

if MNIST == "Training Our Model":
    # Set up logging lists
    epochs = epochs
    epoch_log = []
    loss_log = []
    accuracy_log = []

    # Initialize Streamlit elements for progress tracking
    epoch_progress = tab1.progress(0)

    loss_chart = tab1.line_chart(name="loss_chart")
    accuracy_chart = tab1.line_chart()

    # Training loop
    for epoch in range(epochs):
        MNIST_options.write(f"Starting Epoch: {epoch + 1}...")
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                MNIST_options.write(
                    f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

                # Update charts
                loss_chart.add_rows([{"Loss": actual_loss}])
                accuracy_chart.add_rows([{"Accuracy": accuracy}])

        # Store training stats after each epoch
        epoch_log.append(epoch_num)
        loss_log.append(actual_loss)
        accuracy_log.append(accuracy)
        epoch_progress.progress((epoch + 1) / epochs)

    tab1.write('Finished Training')

    # Save the model's state dictionary
    PATH = './pages/trained_model/mnist_cnn_net.pth'
    torch.save(net.state_dict(), PATH)

    # Load one mini-batch from the test loader
    dataiter = iter(testloader)
    images, labels = next(dataiter)  # Use next() function here


    # Function to display images in Streamlit
    def st_imshow(img):
        img = img / 2 + 0.5  # Unnormalize the image
        npimg = img.numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.axis('off')
        tab1.pyplot(fig)


    # Display the images in the mini-batch
    st_imshow(torchvision.utils.make_grid(images)
              )
    # Print the ground truth labels in Streamlit
    st.sidebar.write('GroundTruth: ', ' '.join('%1s' % labels[j].item() for j in range(len(labels))))


if MNIST == "Reload the model we just saved":
    # Load model and weights
    PATH = './pages/trained_model/mnist_cnn_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)


    # Define a function to display images in Streamlit
    def st_imshow(img, title=""):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.axis('off')
        col.pyplot(fig)


    # Load one mini-batch from the test loader
    test_iter = iter(testloader)
    images, labels = next(test_iter)

    # Move data to GPU
    images = images.to(device)
    labels = labels.to(device)

    # Get model predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Display the predicted results
    st.sidebar.write('Predicted: ', ' '.join('%1s' % predicted[j].cpu().numpy() for j in range(len(predicted))))

    # Calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    tab1.write(f'Accuracy of the network on the 10000 test images: {accuracy:.3f}%')

    # Plot accuracy and loss
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Sample epoch_log and accuracy_log for demonstration purposes
    epoch_log = list(range(1, 11))  # Example epochs
    loss_log = np.random.rand(len(epoch_log))  # Example loss values
    accuracy_log = np.random.rand(len(epoch_log)) * 100  # Example accuracy values

    ax1.plot(epoch_log, loss_log, 'g-')
    ax2.plot(epoch_log, accuracy_log, 'b-')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Test Accuracy', color='b')

    plt.title("Accuracy & Loss vs Epoch")
    plt.xticks(rotation=45)
    tab1.pyplot(fig)

    # Display incorrect predictions
    incorrect_examples = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            predictions = torch.argmax(outputs, dim=1)
            for i in range(images.size(0)):
                if labels[i] != predictions[i]:
                    incorrect_examples.append((images[i].cpu().numpy(), labels[i].item(), predictions[i].item()))

    # Define columns in Streamlit
    num_columns = 4
    cols = tab1.columns(num_columns)

    # Display incorrect predictions in columns
    for idx, (img, true_label, pred_label) in enumerate(incorrect_examples):
        if idx >= len(cols) * 5:  # Limit the number of images displayed
            break
        col = cols[idx % num_columns]
        with col:
            col.write(f'Actual Label: {true_label}, Predicted Label: {pred_label}')
            img = np.reshape(img, [28, 28])
            st_imshow(torch.tensor(img).unsqueeze(0))

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
PyTorch_Regularisation = tab4.selectbox("PyTorch - Fashion-MNSIT Part 1 - No Regularisation", ["Our Data Transform","Building Our Model","Training Our Model","Data Augmentation Example"])
PyTorch_Regularisation_options = tab4.expander("PyTorch - Fashion-MNSIT Part 1 - No Regularisation")
PyTorch_Regularisation_inputs = tab4.expander("Results : ")
placeholder4 = tab4.empty()

# Streamlit title
tab4.title('Fashion MNIST with PyTorch')

# Set up user inputs
epochs = epochs

# Transform to a PyTorch tensor and normalize values between -1 and +1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Define the Neural Network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create containers for logs
epoch_log = []
loss_log = []
accuracy_log = []

# Calculate accuracy on test set
correct = 0
total = 0

# Define data augmentation transforms
data_aug_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15,  interpolation=PIL.Image.BILINEAR),
    transforms.Grayscale(num_output_channels=1)
])

# Function to display augmented images
def show_augmentations(image, augmentations=6):
    fig, ax = plt.subplots(1, augmentations, figsize=(augmentations * 2, 2))
    for i in range(augmentations):
        augmented_img = data_aug_transform(image)
        ax[i].imshow(np.array(augmented_img), cmap='Greys_r')
        ax[i].axis('off')
    tab4.pyplot(fig)

# Load the first image from our training data as a numpy array
image = trainset.data[0].numpy()

# Convert it to PIL image format
img_pil = PIL.Image.fromarray(image)


if PyTorch_Regularisation == "Our Data Transform":


    # Load the Training Data
    tab4.write("Loading Training Data...")
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Load the Test Data
    tab4.write("Loading Test Data...")
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Display some basic dataset info
    tab4.write(f"Training dataset size: {len(trainset)}")
    tab4.write(f"Test dataset size: {len(testset)}")


    # If you want, you could display a few images from the dataset
    def show_images(dataset, num_images=5):
        # Get a batch of images from the DataLoader
        images, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_images)))
        images = images / 2 + 0.5  # unnormalize
        tab4.image(images.permute(0, 2, 3, 1).numpy(), width=100)


    if tab4.button('Show Sample Images'):
        tab4.write("Sample images from the training set:")
        show_images(trainset)


if PyTorch_Regularisation == "Building Our Model":
    # Instantiate the model
    net = Net()
    net.to(device)

    # Display the model architecture
    st.subheader("CNN Model Architecture")
    st.text(net)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    tab4.write(f"Model initialized and moved to {device}.")


if PyTorch_Regularisation == "Training Our Model":
    # Training loop
    if tab4.button('Start Training'):
        tab4.write(f"Training for {epochs} epochs with learning rate {lr} and momentum {momentum}...")
        progress_bar = st.progress(0)

        for epoch in range(epochs):
            tab4.write(f'Starting Epoch: {epoch + 1}...')
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    accuracy = 100 * correct / total
                    epoch_num = epoch + 1
                    actual_loss = running_loss / 100
                    tab4.write(
                        f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                    running_loss = 0.0

            # Update logs
            epoch_log.append(epoch_num)
            loss_log.append(actual_loss)
            accuracy_log.append(accuracy)

            # Update progress bar
            progress_bar.progress((epoch + 1) / epochs)

        tab4.write('Finished Training')
        # Plotting the loss and accuracy
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        ax[0].plot(epoch_log, loss_log, label='Loss')
        ax[0].set_title('Loss over Epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')

        ax[1].plot(epoch_log, accuracy_log, label='Accuracy', color='orange')
        ax[1].set_title('Accuracy over Epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy (%)')

        tab4.pyplot(fig)

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        tab4.write(f'Accuracy of the network on the 10,000 test images: {accuracy:.2f}%')

if PyTorch_Regularisation == "Data Augmentation Example":
    # Input for the number of augmentations to display
    augmentations = st.slider('Number of Augmented Images', min_value=2, max_value=10, value=6)

    # Display the augmented images
    if tab4.button('Show Augmented Images'):
        tab4.write(f"Displaying {augmentations} augmentations of the first image from the dataset.")
        show_augmentations(img_pil, augmentations)


# ----------------------------------------------------------------------------------------------------------------------

