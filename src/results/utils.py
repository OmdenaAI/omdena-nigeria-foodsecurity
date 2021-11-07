
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

@st.cache(allow_output_mutation=True)
def load_model(dataset):
  """
  Loads the models.
  """

  if dataset == 'Maize':
    model = tf.keras.models.load_model('../models/model_maize_final.h5')
  elif dataset == 'Rice':
    model = tf.keras.models.load_model('..models/model_rice.h5')
  elif dataset == 'Crops':
    model = tf.keras.models.load_model('../models/crop_classification.h5')
  return model

def crop_preprocess_img(image, TARGET_SIZE):
  pil_img_rgb = image.convert('RGB') 
  cv2_img = np.array(pil_img_rgb) #convert PIL Image to cv2 format
  img = cv2.resize(cv2_img, (TARGET_SIZE, TARGET_SIZE)) #resize the image to (224, 224)
  img = img_to_array(img) #convert to numpy array
  img = img/255 #normalize the image
  img = np.expand_dims(img, axis = 0) #expands dimension to one sample
  return img

def disease_preprocess_img(image, TARGET_SIZE):
  """ Preprocess the image in the same way the images for model building were processed. 
  
  image: uploaded image

  Returns: image preprocessed using resnet50's preprocess_input
  """

  img = image.resize((TARGET_SIZE, TARGET_SIZE))
  img = img_to_array(img) #convert to numpy array
  img = np.expand_dims(img, axis = 0) #expands dimension to one sample
  img = preprocess_input(img) #preprocess function by resnet50
  return img

def predict(image, model):
  """ Predicts the disease of the image.

  image: preprocessed image
  model: model for maize or rice leaf disease datasets

  Returns: probability for each class/disease
  """
  predictions = model.predict(image)
  return predictions

def get_class(crop, idx):
  """ Gets the disease label for the predicted image.

  crop: crop type - maize or rice
  idx: column index of the class with the highest probability in the predictions

  Returns: disease label

  """
  if crop == 'Maize':
    CLASSES = {
        0: 'Blight',
        1: 'Common Rust',
        2: 'Gray Leaf Spot',
        3: 'Healthy'
    }
  elif crop == 'Rice':
    CLASSES = {
        0: 'Bacterial Leaf Blight',
        1: 'Brown Spot',
        2: 'Leaf Smut'
    }
  elif crop == 'Crops':
    CLASSES = {
        0: 'Rice',
        1: 'Maize'
    }

  return CLASSES[idx]

def get_description(disease):
  """ Gets the disease description

    disease: predicted disease for the image
  """
  maize_blight = f"""
  <p> 
  The tan lesions of northern corn leaf blight are slender and oblong tapering at the ends ranging in size between 1 to 6 inches. </p>
  <p>Lesions run parallel to the leaf margins beginning on the lower leaves and moving up the plant. They may coalesce and cover the enter leaf.
  </p>
  <p>Spores are produced on the underside of the leaf below the lesions giving the appearance of a dusty green fuzz.</p>

  <p> <i>Source: https://cals.cornell.edu/field-crops/corn/diseases-corn/</i></p>
  """

  maize_common_rust = f"""
  <p> Common rust is caused by the fungus Puccinia sorghi. </p>
  <p>Small, round to elongate brown pustules form on both leaf surfaces and other above ground parts of the plant. </p>
  <p>As the pustules mature they become brown to black. If disease is severe, the leaves may yellow and die early
  </p>

  <p> <i>Source: https://cals.cornell.edu/field-crops/corn/diseases-corn/</i></p>
  """
  maize_gray_leaf_spot = f"""
  <p> Gray leaf spot is caused by the fungus Cercospora zeae-maydis. 
  Lesions start as a small dot surrounded by yellow halo, and then will elongate over time parallel to the veins becoming pale brown to gray.
  </p>

  <p> <i>Source: https://cals.cornell.edu/field-crops/corn/diseases-corn/</i></p>
  """

  maize_healthy = f""""""

  rice_bacterial_leaf_blight = f"""
  <p> Rice Bacterial Blight is a deadly bacterial disease that is among the most destructive afflictions of cultivated rice (Oryza sativa and O. glaberrima). 
  In severe epidemics, crop loss may be as high as 75 percent, and millions of hectares of rice are infected annually.
  </p>
  """

  rice_brown_spot = f"""
  <p> Brown spot is caused by the fungus Cochliobolus miyabeanus. Also called Helminthosporium leaf spot, it is one of the most prevalent rice diseases. 
  It is a fungal disease that infects the coleoptile, leaves, leaf sheath, panicle branches, glumes, and spikelets.
  </p>
  """

  rice_leaf_smut = f"""
  <p> Leaf smut, caused by the fungus Entyloma oryzae, is a widely distributed, but somewhat minor, disease of rice. 
  The fungus produces slightly raised, angular, black spots (sori) on both sides of the leaves.
  </p>
  """

  dict_description = {
      'Blight': maize_blight,
      'Common Rust': maize_common_rust,
      'Gray Leaf Spot': maize_gray_leaf_spot,
      'Healthy': maize_healthy,
      'Bacterial Leaf Blight': rice_bacterial_leaf_blight,
      'Brown Spot': rice_brown_spot,
      'Leaf Smut': rice_leaf_smut
  }

  return dict_description[disease]