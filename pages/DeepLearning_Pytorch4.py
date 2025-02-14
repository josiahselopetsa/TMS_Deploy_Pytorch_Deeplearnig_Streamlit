# Import PyTorch
import torch

# We use torchvision to get our dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms
import streamlit as st
import cv2

# Display the first image
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# Import PyTorch's optimization libary and nn
# nn is used as the basic building block for our Network graphs
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix, classification_report

from PIL import Image

from matplotlib import cm


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
tab3,tab5 = st.tabs(["35. CycleGAN - Turn Horses into Zebras","39. Facial Recognition with VGGFace in Keras"])  # ,"G","R","Hue"

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
# "35. CycleGAN - Turn Horses into Zebras"
CycleGAN = tab3.selectbox("CycleGAN - Turn Horses into Zebras", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
CycleGAN_options = tab3.expander("CycleGAN - Turn Horses into Zebras")
CycleGAN_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# "39. Facial Recognition with VGGFace in Keras"
VGGFace = tab5.selectbox("Facial Recognition with VGGFace in Keras", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
VGGFace_options = tab5.expander("Facial Recognition with VGGFace in Keras")
VGGFace_inputs = tab5.expander("Results : ")
placeholder5 = tab5.empty()
