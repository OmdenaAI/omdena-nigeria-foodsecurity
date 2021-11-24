
# Crop Disease Detection

## Summary

Plant diseases have negative effects on the crop production, and if those diseases are not detected on time, there will be an increase in the food insecurity. Food and Agriculture Organization (FAO) estimates that annually between 20-40% of global crop production is lost to pests. Each year, plant diseases cost the global economy over 200bn dollars. In addition, farmers rely on their naked eye observation which can be a complex task because of the similarities in the color and shape of the plant diseases. This leads some of them to hire experts for consultation which might be too expensive for them. Hence, many researchers studied the automatic detection of plant diseases using Artificial Intelligence. In this task, the collaborators used Transfer Learning and ResNet50 to build the model for crop disease detection. The collected data was limited only in rice and maize leaf diseases. Dropout Regularization and Image augmentation techniques like shifting, rotation, flipping, zooming, and adjusting brightness were applied to combat overfitting. The models for maize and rice leaf disease detection performed very well on the test data with 91% and 92% accuracy respectively.

## About the data

The images for maize and rice were collected from Kaggle. Maize dataset is imbalanced with one class containing number of images half of the other classes. Rice has a very small dataset with 40 images only for each disease.

Maize - Imbalanced Dataset            |  Rice - Small Dataset
:-------------------------:|:-------------------------:
![](https://github.com/OmdenaAI/omdena-nigeria-foodsecurity/blob/main/src/tasks/task-3-crop-disease-detection/images/maize_imbalanced.PNG?raw=true)  |  ![](https://github.com/OmdenaAI/omdena-nigeria-foodsecurity/blob/main/src/tasks/task-3-crop-disease-detection/images/rice_small_dataset.PNG?raw=true)

## Data Preprocessing

The built-in function of ResNet50's preprocessing function is used to convert images into arrays. Since we had a very small dataset for rice and an imbalanced dataset for maize, the model might learn the patterns and noise in the data to the extent that it negatively impacts the model performance in generalizing new data. This is called overfitting. In this task, image augmentation was applied to increase the number of the training samples as one way to overcome overfitting.

![](https://github.com/OmdenaAI/omdena-nigeria-foodsecurity/blob/main/src/tasks/task-3-crop-disease-detection/images/image_augmentation.png?raw=true)

## Model Building and Performance
Transfer Learning was used to transfer the knowledge of ResNet50 to our own task which is detecting the maize and rice leaf diseases. ResNet50 stands for Residual Networks with 50 layers. It is a pretrained model that is trained on a million images from the Imagenet database and classifies 1000 classes.

Here is the model architecture of ResNet50.
<!--insert image here-->
![](https://github.com/OmdenaAI/omdena-nigeria-foodsecurity/blob/main/src/tasks/task-3-crop-disease-detection/images/resnet50.png?raw=true)

We replaced the top layer that classifies 1000 classes with our very own classifier layer with 3 outputs. We added a dropout layer to combat overfitting. Dropout regularization is a popular technique to regularize neural networks. Adding a dropout layer would exclude a fraction of neurons along with its incoming and outgoing connections, so that they won’t contribute much information during those updates, which forces other active nodes to learn harder and thus reduces the error. In our model, each neuron has a 40% chance of being removed in the network.
<!-- insert image here -->
![](https://github.com/OmdenaAI/omdena-nigeria-foodsecurity/blob/main/src/tasks/task-3-crop-disease-detection/images/dropout.png?raw=true)

For maize model, another technique that we tried to combat overfitting was to set the weights for each class so that the model will pay more attention to the minority class by giving its higher weight. We used scikit learn's compute_class_weight function for this task.

We tried to made our own CNN model but it didn't perform well. We have tried also the other pretrained models such as VGG16 and Xception but ResNet50 outperformed them. See the model's performance for maize leaf disease detection below.

Model            |  Accuracy            |  F1-Score
:-------------------------:|:-------------------------:|:-------------------------:
VGG16  |  0.90  |  0.88
Xception  |  0.86  |  0.83
ResNet50  |  0.91  |  0.91
3-layer CNN  |  0.79  |  0.79

For rice model, we have explored ResNet50 only. The model correctly classified 92% of the test data.

## Resource Links

1. https://www.kaggle.com/vbookshelf/rice-leaf-diseases?select=rice_leaf_diseases
2. https://www.kaggle.com/smaranjitghose/corn-or-maize-leaf-disease-dataset
3. https://www.researchgate.net/publication/255483247_Automatic_plant_pest_detection_recognition_using_k-means_clustering_algorithm_correspondence_filters
4. https://keras.io/api/applications/resnet/#resnet50-function
5. https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
6. https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99
