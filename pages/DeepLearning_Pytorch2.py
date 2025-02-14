# Import PyTorchR
import torch
import json
import streamlit as st
import cv2

# Display the first image
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from matplotlib import cm

import torchvision.models as models
from torchsummary import summary
from torchvision import transforms
import io
import torch.nn.functional as nnf
from os import listdir
from os.path import isfile, join
from torch.autograd import Variable

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
tab2,tab3,tab4,tab5 = st.tabs([ "13. PyTorch Pretrained Models - 1 - VGG16, ResNet, Inceptionv3, MobileNetv2, SqueezeNet, WideResNet and MNASNet","15. PyTorch - Rank-1 and Rank-5 Accuracy", "17. Cats vs Dogs PyTorch","19. PyTorch Lightening Tutorial - Batch and LR Selection, Tensorboards, Callbacks, mGPU, TPU and more"])  # ,"G","R","Hue"

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
# "13. PyTorch Pretrained Models - 1 - VGG16, ResNet, Inceptionv3, MobileNetv2, SqueezeNet, WideResNet and MNASNet"

Pretrained_Models  = tab2.selectbox("13. PyTorch Pretrained Models - 1 - VGG16, ResNet, Inceptionv3, MobileNetv2, SqueezeNet, WideResNet and MNASNet",
                                    ["Loading VGG16", "Loading ResNet","Loading Inception","Loading MobileNet", "Loading SqueezeNet","Loading Wide ResNet","Loading Wide MNASNet"])
Pretrained_Models_options = tab2.expander("13. PyTorch Pretrained Models - 1 - VGG16, ResNet, Inceptionv3, MobileNetv2, SqueezeNet, WideResNet and MNASNet")
Pretrained_Models_inputs = tab2.expander("Results : ")
placeholder2 = tab2.empty()

if Pretrained_Models == "Loading VGG16":
    # Title of the Streamlit App
    Pretrained_Models_options.title('VGG16 Model Summary')

    # Load the pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))

    # Use torchsummary to display a summary of the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Move the model to the selected device
    model.to(device)

    # Display the summary in the main panel
    Pretrained_Models_options.subheader("Model Summary")
    input_size = (3, 224, 224)

    # Capture summary output in a string
    from io import StringIO
    import sys


    def capture_model_summary(model, input_size):
        buffer = StringIO()
        sys.stdout = buffer
        summary(model, input_size=input_size)
        sys.stdout = sys.__stdout__
        return buffer.getvalue()


    model_summary = capture_model_summary(model, input_size)

    # Display summary in Streamlit
    Pretrained_Models_options.text(model_summary)

    # Define transforms (you can add normalization if needed)
    Pretrained_Models_options.subheader("Test Transforms")
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Uncomment if normalization is needed
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    Pretrained_Models_options.text(test_transforms)

    # Set model to evaluation mode
    model.eval()
    Pretrained_Models_options.write("The model is now in evaluation mode.")


    # Load the ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)


    class_names = load_class_names()

    # Define image transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Uncomment below if normalization is needed
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict image class
    def predict_image(image, model, class_names):
        image_tensor = test_transforms(image).unsqueeze_(0).to(device).float()
        output = model(image_tensor)
        index = output.data.cpu().numpy().argmax()
        class_name = class_names[str(index)]
        return class_name


    # Streamlit App
    tab2.title('Image Classification with VGG16')

    # File uploader for images
    uploaded_files = tab2.file_uploader("Upload images VGG16", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Display and predict each uploaded image
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')

            tab2.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            prediction = predict_image(image, model, class_names)
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator

    tab2.write("Upload an image to classify using the VGG16 model!")


if Pretrained_Models == "Loading ResNet":
    # Title for the app
    Pretrained_Models_options.title("ResNet18 Model Summary")

    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))

    # Set the model to evaluation mode
    model.eval()
    Pretrained_Models_options.write("The model is set to evaluation mode.")


    # Model summary function
    @st.cache_data
    def get_model_summary(_model, input_size):
        from io import StringIO
        import sys

        # Redirect print output to capture the summary
        buffer = StringIO()
        sys.stdout = buffer
        summary(_model, input_size=input_size)
        sys.stdout = sys.__stdout__

        return buffer.getvalue()


    # Show the model summary
    input_size = (3, 224, 224)
    Pretrained_Models_options.subheader("Model Summary")
    model_summary = get_model_summary(model, input_size)
    Pretrained_Models_options.text(model_summary)


    # Load ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)


    class_names = load_class_names()

    # Define image transformations (Resize, ToTensor, etc.)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Optionally, normalize the image if needed for your model
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict the class of an image
    def predict_image(image, model, class_names):
        # Apply transformations and prepare the image
        image_tensor = test_transforms(image).unsqueeze_(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            index = output.data.cpu().numpy().argmax()
            class_name = class_names[str(index)]

        return class_name


    # Function to get images from the user input (via file uploader)
    def get_images(uploaded_files):
        images = []
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            images.append(img)
        return images


    # Streamlit App Layout
    tab2.title('Image Classification with ResNet18')

    # Upload images using Streamlit's file uploader
    uploaded_files = tab2.file_uploader("Upload Images ResNet18", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process and predict the images
    if uploaded_files:
        images = get_images(uploaded_files)

        tab2.subheader('Predictions')
        for i, image in enumerate(images):
            # Display the uploaded image
            tab2.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

            # Make a prediction
            prediction = predict_image(image, model, class_names)

            # Display the prediction result
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator between images


if Pretrained_Models == "Loading Inception":
    # Title for the Streamlit app
    Pretrained_Models_options.title("InceptionV3 Model Summary")

    # Load the pre-trained InceptionV3 model
    model = models.inception_v3(pretrained=True)

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))

    # Set the model to evaluation mode
    model.eval()
    Pretrained_Models_options.write("The model is set to evaluation mode.")


    # Model summary function
    @st.cache_data
    def get_model_summary(_model, input_size):
        from io import StringIO
        import sys

        # Redirect print output to capture the summary
        buffer = StringIO()
        sys.stdout = buffer
        summary(_model, input_size=input_size)
        sys.stdout = sys.__stdout__

        return buffer.getvalue()


    # Input size for InceptionV3
    input_size = (3, 299, 299)

    # Show the model summary
    Pretrained_Models_options.subheader("Model Summary")
    model_summary = get_model_summary(model, input_size)
    Pretrained_Models_options.text(model_summary)


    # Load ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)


    class_names = load_class_names()

    # Define image transformations (Resize, ToTensor, etc.)
    test_transforms = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299 input size
        transforms.ToTensor(),
        # Optionally, normalize the image if needed for your model
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict the class of an image
    def predict_image(image, model, class_names):
        # Apply transformations and prepare the image
        image_tensor = test_transforms(image).unsqueeze_(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            index = output.data.cpu().numpy().argmax()
            class_name = class_names[str(index)]

        return class_name


    # Streamlit App Layout
    tab2.title('Image Classification with InceptionV3')

    # Upload images using Streamlit's file uploader
    uploaded_files = tab2.file_uploader("Upload Images InceptionV3", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process and predict the images
    if uploaded_files:
        st.subheader('Predictions')

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')

            # Display the uploaded image
            tab2.image(image, caption="Uploaded Image", use_column_width=True)

            # Make a prediction
            prediction = predict_image(image, model, class_names)

            # Display the prediction result
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator between images

if Pretrained_Models == "Loading MobileNet":
    # Title for the Streamlit app
    Pretrained_Models_options.title("MobileNetV2 Model Summary")


    # Load the pre-trained MobileNetV2 model
    model = models.mobilenet_v2(pretrained=True)

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))

    # Set the model to evaluation mode
    model.eval()
    Pretrained_Models_options.write("The model is set to evaluation mode.")


    # Model summary function
    @st.cache_data
    def get_model_summary(_model, input_size):
        from io import StringIO
        import sys

        # Redirect print output to capture the summary
        buffer = StringIO()
        sys.stdout = buffer
        summary(_model, input_size=input_size)
        sys.stdout = sys.__stdout__

        return buffer.getvalue()


    # Input size for MobileNetV2
    input_size = (3, 224, 224)

    # Show the model summary
    Pretrained_Models_options.subheader("Model Summary")
    model_summary = get_model_summary(model, input_size)
    Pretrained_Models_options.text(model_summary)


    # Load ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)


    class_names = load_class_names()

    # Define image transformations (Resize, ToTensor, etc.)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224 input size
        transforms.ToTensor(),
        # Optionally, normalize the image if needed for your model
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict the class of an image
    def predict_image(image, model, class_names):
        # Apply transformations and prepare the image
        image_tensor = test_transforms(image).unsqueeze_(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            index = output.data.cpu().numpy().argmax()
            class_name = class_names[str(index)]

        return class_name


    # Function to get images from the user input (via file uploader)
    def get_images(uploaded_files):
        images = []
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            images.append(img)
        return images


    # Streamlit App Layout
    tab2.title('Image Classification with MobileNetV2')

    # Upload images using Streamlit's file uploader
    uploaded_files = tab2.file_uploader("Upload Images MobileNetV2", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process and predict the images
    if uploaded_files:
        images = get_images(uploaded_files)

        st.subheader('Predictions')
        for i, image in enumerate(images):
            # Display the uploaded image
            tab2.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

            # Make a prediction
            prediction = predict_image(image, model, class_names)

            # Display the prediction result
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator between images

if Pretrained_Models == "Loading SqueezeNet":

    # Load the pre-trained SqueezeNet1_0 model
    model = models.squeezenet1_0(pretrained=True).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Streamlit App Layout
    Pretrained_Models_options.title("SqueezeNet1_0 Model Summary")

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))


    # Model summary function
    @st.cache_data
    def get_model_summary(_model, input_size):
        from io import StringIO
        import sys

        # Redirect print output to capture the summary
        buffer = StringIO()
        sys.stdout = buffer
        summary(_model, input_size=input_size)
        sys.stdout = sys.__stdout__

        return buffer.getvalue()


    # Input size for SqueezeNet1_0
    input_size = (3, 224, 224)  # SqueezeNet1_0 expects 224x224 input size

    # Show the model summary
    Pretrained_Models_options.subheader("Model Summary")
    model_summary = get_model_summary(model, input_size)
    Pretrained_Models_options.text(model_summary)


    # Load ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)


    class_names = load_class_names()

    # Define image transformations (Resize, ToTensor, etc.)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # SqueezeNet1_0 expects 224x224 input size
        transforms.ToTensor(),
        # Optionally, normalize the image if needed for your model
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict the class of an image
    def predict_image(image, model, class_names):
        # Apply transformations and prepare the image
        image_tensor = test_transforms(image).unsqueeze_(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            index = output.data.cpu().numpy().argmax()
            class_name = class_names[str(index)]

        return class_name


    # Function to get images from the user input (via file uploader)
    def get_images(uploaded_files):
        images = []
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            images.append(img)
        return images


    # Streamlit App Layout
    tab2.title('Image Classification with SqueezeNet1_0')

    # Upload images using Streamlit's file uploader
    uploaded_files = tab2.file_uploader("Upload Images SqueezeNet1_0", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process and predict the images
    if uploaded_files:
        images = get_images(uploaded_files)

        st.subheader('Predictions')
        for i, image in enumerate(images):
            # Display the uploaded image
            tab2.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

            # Make a prediction
            prediction = predict_image(image, model, class_names)

            # Display the prediction result
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator between images


if Pretrained_Models == "Loading Wide ResNet":

    # Load the pre-trained WideResNet50_2 model
    model = models.wide_resnet50_2(pretrained=True).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Streamlit App Layout
    Pretrained_Models_options.title("WideResNet50_2 Model Summary")

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))


    # Model summary function
    @st.cache_data
    def get_model_summary(_model, input_size):
        from io import StringIO
        import sys

        # Redirect print output to capture the summary
        buffer = StringIO()
        sys.stdout = buffer
        summary(_model, input_size=input_size)
        sys.stdout = sys.__stdout__

        return buffer.getvalue()


    # Input size for WideResNet50_2
    input_size = (3, 224, 224)  # WideResNet50_2 expects 224x224 input size

    # Show the model summary
    Pretrained_Models_options.subheader("Model Summary")
    model_summary = get_model_summary(model, input_size)
    Pretrained_Models_options.text(model_summary)


    # Load ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)


    class_names = load_class_names()

    # Define image transformations (Resize, ToTensor, etc.)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # WideResNet50_2 expects 224x224 input size
        transforms.ToTensor(),
        # Optionally, normalize the image if needed for your model
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict the class of an image
    def predict_image(image, model, class_names):
        # Apply transformations and prepare the image
        image_tensor = test_transforms(image).unsqueeze_(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            index = output.data.cpu().numpy().argmax()
            class_name = class_names[str(index)]

        return class_name


    # Function to get images from the user input (via file uploader)
    def get_images(uploaded_files):
        images = []
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            images.append(img)
        return images


    # Streamlit App Layout
    tab2.title('Image Classification with WideResNet50_2')

    # Upload images using Streamlit's file uploader
    uploaded_files = tab2.file_uploader("Upload Images WideResNet50_2", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process and predict the images
    if uploaded_files:
        images = get_images(uploaded_files)

        tab2.subheader('Predictions')
        for i, image in enumerate(images):
            # Display the uploaded image
            tab2.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

            # Make a prediction
            prediction = predict_image(image, model, class_names)

            # Display the prediction result
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator between images

if Pretrained_Models == "Loading Wide MNASNet":
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained MNASNet1_0 model
    model = models.mnasnet1_0(pretrained=True).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Streamlit App Layout
    Pretrained_Models_options.title("MNASNet1_0 Model Summary")

    # Display model architecture
    Pretrained_Models_options.subheader("Model Architecture")
    Pretrained_Models_options.text(str(model))


    # Model summary function
    @st.cache_data
    def get_model_summary(_model, input_size):
        from io import StringIO
        import sys

        # Redirect print output to capture the summary
        buffer = StringIO()
        sys.stdout = buffer
        summary(_model, input_size=input_size)
        sys.stdout = sys.__stdout__

        return buffer.getvalue()


    # Input size for MNASNet1_0
    input_size = (3, 224, 224)  # MNASNet1_0 expects 224x224 input size

    # Show the model summary
    Pretrained_Models_options.subheader("Model Summary")
    model_summary = get_model_summary(model, input_size)
    Pretrained_Models_options.text(model_summary)


    # Load ImageNet class labels
    @st.cache_data
    def load_class_names():
        with open('./pages/imageNetclasses.json') as f:
            return json.load(f)

    class_names = load_class_names()

    # Define image transformations (Resize, ToTensor, etc.)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # MNASNet1_0 expects 224x224 input size
        transforms.ToTensor(),
        # Optionally, normalize the image if needed for your model
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Function to predict the class of an image
    def predict_image(image, model, class_names):
        # Apply transformations and prepare the image
        image_tensor = test_transforms(image).unsqueeze_(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            index = output.data.cpu().numpy().argmax()
            class_name = class_names[str(index)]

        return class_name


    # Function to get images from the user input (via file uploader)
    def get_images(uploaded_files):
        images = []
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            images.append(img)
        return images


    # Streamlit App Layout
    tab2.title('Image Classification with MNASNet1_0')

    # Upload images using Streamlit's file uploader
    uploaded_files = tab2.file_uploader("Upload Images MNASNet1_0", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process and predict the images
    if uploaded_files:
        images = get_images(uploaded_files)

        tab2.subheader('Predictions')
        for i, image in enumerate(images):
            # Display the uploaded image
            st.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

            # Make a prediction
            prediction = predict_image(image, model, class_names)

            # Display the prediction result
            tab2.write(f"Predicted Label: **{prediction}**")

            tab2.write("---")  # Separator between images
# ----------------------------------------------------------------------------------------------------------------------
# "15. PyTorch - Rank-1 and Rank-5 Accuracy"
Rank = tab3.selectbox("PyTorch - Rank-1 and Rank-5 Accuracy",
                      ["Load VGG16", "Get our Class Probabilities","Construct our function to give us our Rank-N Accuracy"])
Rank_options = tab3.expander("PyTorch - Rank-1 and Rank-5 Accuracy")
Rank_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()

# Load the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Define the transformation to be applied to the image
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load class names
with open('./pages/imageNetclasses.json') as f:
    class_names = json.load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if Rank == "Load VGG16":

    def predict(image):
        # Apply transformations
        image_tensor = test_transforms(image).float().unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Make predictions
        with torch.no_grad():
            output = model(image_tensor)

        # Get predicted class
        index = output.data.cpu().numpy().argmax()
        name = class_names[str(index)]
        return name


    tab3.title("VGG16 Image Classifier")

    # Upload an image
    uploaded_file = tab3.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        tab3.image(image, caption='Uploaded Image.', use_column_width=True)

        # Predict and display the result
        class_name = predict(image)
        tab3.write(f'Predicted class: {class_name}')

        # Plot image with prediction title
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        ax.set_title(f'Predicted: {class_name}')
        ax.imshow(image)

        # Save plot to a BytesIO object and display it in Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        tab3.image(buf, caption='Prediction Plot', use_column_width=True)

if Rank == "Get our Class Probabilities":

    def predict(image):
        # Apply transformations
        image_tensor = test_transforms(image).float().unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Make predictions
        with torch.no_grad():
            output = model(image_tensor)

        # Get probabilities and top 5 predictions
        prob = nnf.softmax(output, dim=1)
        top_p, top_class = prob.topk(5, dim=1)

        top_p = top_p.cpu().data.numpy()[0]
        top_class = top_class.cpu().data.numpy()[0]

        return top_p, top_class


    def get_class_names(top_classes):
        return [class_names[str(c)] for c in top_classes]

    tab3.title("VGG16 Image Classifier")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        tab3.image(image, caption='Uploaded Image.', use_column_width=True)

        # Predict and display the results
        top_p, top_class = predict(image)
        class_names_list = get_class_names(top_class)

        tab3.write('Top 5 Predictions:')
        for i in range(5):
            tab3.write(f'{class_names_list[i]}: {top_p[i] * 100:.2f}%')

        # Plot image with prediction title
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        ax.set_title('Top 5 Predictions')
        ax.imshow(image)

        # Save plot to a BytesIO object and display it in Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        tab3.image(buf, caption='Prediction Plot', use_column_width=True)


if Rank == "Construct our function to give us our Rank-N Accuracy":

    def get_class_names(top_classes):
        # Convert tensor indices to a list of string class names
        top_classes = top_classes.cpu().data.numpy()[0]  # Convert to numpy array and flatten
        return [class_names[str(index)] for index in top_classes]


    def get_rank_n(directory, ground_truth, N, show_images=True):
        # Get image names in directory
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

        all_top_classes = []

        for (i, image_filename) in enumerate(onlyfiles):
            image = Image.open(join(directory, image_filename))

            # Convert to Tensor
            image_tensor = test_transforms(image).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor).to(device)
            output = model(input)

            # Get probabilities and top-N class names
            prob = nnf.softmax(output, dim=1)
            top_p, top_class = prob.topk(N, dim=1)
            top_class_names = get_class_names(top_class)
            all_top_classes.append(top_class_names)

            if show_images:
                # Plot image
                tab3.image(image, caption=f"Top {N} Predictions: {', '.join(top_class_names)}", use_column_width=True)

        return get_score(all_top_classes, ground_truth, N)


    def get_score(all_top_classes, ground_truth, N):
        in_labels = sum(1 for i, labels in enumerate(all_top_classes) if ground_truth[i] in labels)
        return f'Rank-{N} Accuracy = {in_labels / len(all_top_classes) * 100:.2f}%'


    tab3.title("Rank-N Prediction Accuracy")

    # Directory and ground truth input
    directory = tab3.text_input('Directory containing images:', './pages/images_VGG16/images/class1/')
    ground_truth_input = tab3.text_area('Ground truth labels (one per line):',
                                      'basketball\nGerman shepherd, German shepherd dog, German police dog, alsatian\nlimousine, limo\nspider web, spider\'s web\nburrito\nbeer_glass\ndoormat, welcome mat\nChristmas stocking\ncollie')
    N = tab3.slider('Top-N predictions:', min_value=1, max_value=10, value=5)

    if tab3.button('Calculate Rank-N Accuracy'):
        # Prepare ground truth
        ground_truth = ground_truth_input.split('\n')

        # Calculate and display Rank-N accuracy
        accuracy = get_rank_n(directory, ground_truth, N)
        tab3.write(accuracy)


# ----------------------------------------------------------------------------------------------------------------------
# "17. Cats vs Dogs PyTorch"
Cats_Dogs = tab4.selectbox("Cats vs Dogs PyTorch", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Cats_Dogs_options = tab4.expander("Cats vs Dogs PyTorch")
Cats_Dogs_inputs = tab4.expander("Results : ")
placeholder4 = tab4.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "19. PyTorch Lightening Tutorial - Batch and LR Selection, Tensorboards, Callbacks, mGPU, TPU and more"
Lightening = tab5.selectbox("PyTorch Lightening Tutorial - Batch and LR Selection, Tensorboards, Callbacks, mGPU, TPU and more", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
Lightening_options = tab5.expander("PyTorch Lightening Tutorial - Batch and LR Selection, Tensorboards, Callbacks, mGPU, TPU and more")
Lightening_inputs = tab5.expander("Results : ")
placeholder5 = tab5.empty()


