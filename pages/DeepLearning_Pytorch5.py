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
tab1,tab2,tab3,tab4,tab5 = st.tabs(["41. DeepFace - Age, Gender, Expression, Headpose and Recognition", "43. Object Detection - Mask Detection - TensorFlow Object Detection - MobileNetV2 SSD","45. Object Detection - Pot Hole Detection with TinyYOLOv4", "47. Object Detection - Website Screenshot Region Detection - YOLOv4-Darknet","49. Object Detection - Chess Pieces Detection - YOLOv3 PyTorch"])  # ,"G","R","Hue"

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
# "41. DeepFace - Age, Gender, Expression, Headpose and Recognition"
DeepFace = tab1.selectbox("DeepFace - Age, Gender, Expression, Headpose and Recognition", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
DeepFace_options = tab1.expander("DeepFace - Age, Gender, Expression, Headpose and Recognition")
DeepFace_inputs = tab1.expander("Results : ")
placeholder1 = tab1.empty()

# ----------------------------------------------------------------------------------------------------------------------
# "43. Object Detection - Mask Detection - TensorFlow Object Detection - MobileNetV2 SSD"
MobileNetV2_SSD  = tab2.selectbox("Object Detection - Mask Detection - TensorFlow Object Detection - MobileNetV2 SSD", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
MobileNetV2_SSD_options = tab2.expander("Object Detection - Mask Detection - TensorFlow Object Detection - MobileNetV2 SSD")
MobileNetV2_SSD_inputs = tab2.expander("Results : ")
placeholder2 = tab2.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "45. Object Detection - Pot Hole Detection with TinyYOLOv4"
TinyYOLOv4 = tab3.selectbox("Object Detection - Pot Hole Detection with TinyYOLOv4", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
TinyYOLOv4_options = tab3.expander("Object Detection - Pot Hole Detection with TinyYOLOv4")
TinyYOLOv4_inputs = tab3.expander("Results : ")
placeholder3 = tab3.empty()
# ----------------------------------------------------------------------------------------------------------------------
#  "47. Object Detection - Website Screenshot Region Detection - YOLOv4-Darknet"
YOLOv4_Darknet = tab4.selectbox("Object Detection - Website Screenshot Region Detection - YOLOv4-Darknet", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
YOLOv4_Darknet_options = tab4.expander("Object Detection - Website Screenshot Region Detection - YOLOv4-Darknet")
YOLOv4_Darknet_inputs = tab4.expander("Results : ")
placeholder4 = tab4.empty()
# ----------------------------------------------------------------------------------------------------------------------
# "49. Object Detection - Chess Pieces Detection - YOLOv3 PyTorch"
YOLOv3_PyTorch = tab5.selectbox("Object Detection - Chess Pieces Detection - YOLOv3 PyTorch", ["Our Data Transform", "Defining Our Model","Building Our Model","Training Our Model","Data Augmentation Example"])
YOLOv3_PyTorch_options = tab5.expander("Object Detection - Chess Pieces Detection - YOLOv3 PyTorch")
YOLOv3_PyTorch_inputs = tab5.expander("Results : ")
placeholder5 = tab5.empty()
