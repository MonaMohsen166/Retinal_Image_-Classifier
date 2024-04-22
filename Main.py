from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from tensorflow import keras
from keras.applications import VGG16 
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import normalize 
import matplotlib
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, uic
import sys
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import os
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap  # load image
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QShortcut, QSizePolicy, QSlider,
                             QStyle, QVBoxLayout, QWidget)
from PyQt5.uic import loadUiType  # loadUiType: Open File
from PyQt5.uic import loadUi

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('AdvImg.ui', self) # Load the .ui file
        self.pushButton.clicked.connect(self.Startcode)
        self.show() # Show the GUI

    def Handle_UI(self):
        pass

    def Startcode(self):

        # Define a dictionary to map class names to labels
        class_dict = {'Drusen': 0, 'Exudates': 1, 'Normal': 2}

        # Define the path to the main dataset folder
        dataset_path = "ORNL"
        # dataset_path=self.label_10.text()
        # Load the dataset
        self.label_10.setText(dataset_path)
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for class_name in class_dict.keys(): #loop over folders
            class_path = os.path.join(dataset_path, class_name)
            filenames = os.listdir(class_path)
            n_files = len(filenames)
            n_train = int(0.8 * n_files)
            train_filenames = filenames[:n_train]
            test_filenames = filenames[n_train:]

            for filename in train_filenames: #loop over train
                image_path = os.path.join(class_path, filename)
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                X_train.append(np.array(img))
                y_train.append(class_dict[class_name])
            
            #resize
            for filename in test_filenames: #loop over test
                image_path = os.path.join(class_path, filename)
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                X_test.append(np.array(img))
                y_test.append(class_dict[class_name])
        self.widget_3.canvas.axes.clear()
        self.widget_3.canvas.axes.imshow(img)
        self.widget_3.canvas.draw()
        self.label_46.setText(str(class_name))    
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42) #20% of data metsaba lel testing 
        #y_val has true values #x_val has images without score

        # Load the VGG16 model
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Preprocess the images
        X_train = tf.keras.applications.vgg16.preprocess_input(X_train)
        X_val = tf.keras.applications.vgg16.preprocess_input(X_val)
        X_test = tf.keras.applications.vgg16.preprocess_input(X_test)

        # Extract features from the VGG16 model
        features_train = vgg_model.predict(X_train)
        features_val = vgg_model.predict(X_val)
        features_test = vgg_model.predict(X_test)


        ##########################Predict on Train Data:#############################################################

        # Reshape the features_train array
        n_samples_train = features_train.shape[0]
        features_train_2d = features_train.reshape(n_samples_train, -1)



        ## Train the SVM model
        svm = SVC()
        svm.fit(features_train_2d, y_train) #features train coming from x 
        # y_train has true vales for images that I train on 
        #features_train_2D has images with no true value

        # Reshape the features_test array
        n_samples_test = features_test.shape[0]
        features_test_2d = features_test.reshape(n_samples_test, -1)

        # Make predictions using the SVM model
        predictions_test = svm.predict(features_test_2d) 

        #y_test is images metsaba for testing 
        y_test=np.array(y_test)
        print("y_test len:",len(y_test))
        print("y_test:",y_test)
        predictions_test=np.array(predictions_test)
        print("predicted test:",predictions_test)
        self.label_18.setText(str(len(y_test)) )
        self.label_19.setText(str(y_test) )
        self.label_20.setText(str(predictions_test) )

        confusion_mat = confusion_matrix(y_test, predictions_test)
        accuracy = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat) * 100
        print("Accuracy using correct prediction:",accuracy)
        self.label_6.setText(str(round(accuracy,3)) )
        accuracy = accuracy_score(y_test, predictions_test)
        print("accuracy sklearn:",accuracy)
        self.label_22.setText(str(round(accuracy,3)) )

        # Calculate the precision
        precision = precision_score(y_test, predictions_test, average=None)
        print('Precision: ',precision)
        #LASA
        self.label_31.setText(str(round(precision[0],3)) )
        self.label_34.setText(str(round(precision[1],3)) )
        self.label_35.setText(str(round(precision[2],3)) )

        #Calculate F1 Score
        f1_each=f1_score(y_test, predictions_test, average=None)
        print('F1 score each: ',f1_each)
        self.label_41.setText(str(round(f1_each[0],3)) )
        self.label_44.setText(str(round(f1_each[1],3)) )
        self.label_45.setText(str(round(f1_each[2],3) ))

        recall = recall_score(y_test, predictions_test, average=None)
        print("recall sklearn:",recall)
        self.label_28.setText(str(round(recall[0],3)) )
        self.label_38.setText(str(round(recall[1],3)) )
        self.label_39.setText(str(round(recall[2],3)) )

        #specificty
        def specificity(y_true, y_pred, class_index):
            cm = confusion_matrix(y_true, y_pred)
            tn = sum(cm[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1]) if i != class_index and j != class_index)
            fp = sum(cm[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1]) if i != class_index and j == class_index)
            return tn / (tn + fp)

        Drusen_Specificty_Score = specificity(y_test, predictions_test, 0)
        print("Specificity for class  Drusen: {:.2f}".format(Drusen_Specificty_Score))
        self.label_4.setText(str(round(Drusen_Specificty_Score,3) ))

        Exudate_Specificty_Score = specificity(y_test, predictions_test, 1)
        print("Specificity for class Exudate: {:.2f}".format(Exudate_Specificty_Score))
        self.label_25.setText(str(round(Exudate_Specificty_Score,3) ))

        Normal_Specificty_Score = specificity(y_test, predictions_test, 2)
        print("Specificity for class Normal: {:.2f}".format(Normal_Specificty_Score))
        self.label_26.setText(str(round(Normal_Specificty_Score,3)) )


        #calculate the AUC 
        test = np.reshape(y_test, (-1, 1))
        predict = np.reshape(predictions_test, (-1, 1))
        predict = normalize(predict, axis=1, norm='l1')
        test = normalize(test, axis=1, norm='l1')
        auc = roc_auc_score(test, predict, multi_class='ovo')
        print('AUC: ',auc)
        self.label_14.setText(str(round(auc,3)) )


        # Map predicted class labels to class names
        class_labels = {0: 'Drusen', 1: 'Exudates', 2: 'Normal'}
        predicted_class_labels = [class_labels[label] for label in predictions_test]
        spec_list = []
        test_class_labels = [class_labels[label] for label in y_test]

        # Print the predicted class labels for each test image
        for i in range(len(predicted_class_labels)):
            print(f"Image {i+1}: {predicted_class_labels[i]}")
        self.listWidget.insertItems(0,test_class_labels)
        self.listWidget_2.insertItems(0,predicted_class_labels)
        self.label_48.setText(str(predicted_class_labels[32]) )
        self.label_46.setText(str(class_name))  

    
        sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu")
        

        fig, ax = plt.subplots()

        # Create the heatmap with annotations
        im = ax.imshow(confusion_mat, cmap="YlGnBu")

        # Loop over data dimensions and create text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, confusion_mat[i, j], ha="center", va="center", color="black")

        # Set labels for the axes
        ax.set_title("Confusion Matrix Heatmap for tested data")
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")

        # Clear the previous plot
        self.widget_2.canvas.axes.clear()

        # Display the heatmap with numbers on the widget
        self.widget_2.canvas.axes.imshow(confusion_mat, cmap="YlGnBu")

        # Loop over data dimensions and create text annotations on the widget
        for i in range(3):
            for j in range(3):
                self.widget_2.canvas.axes.text(j, i, confusion_mat[i, j], ha="center", va="center", color="black")

        self.widget_2.canvas.axes.set_title("Confusion Matrix Heatmap for tested data")
        self.widget_2.canvas.axes.set_xlabel("Predicted labels")
        self.widget_2.canvas.axes.set_ylabel("True labels")
        self.widget_2.canvas.draw()
        print("Accuracy using confusion matrix:", accuracy)
 
        
app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
#  app = QApplication(sys.argv)
# window = MainApp ()
# window.show()
# app.exec_()