'''
COSC 311
Homework 2 Code
Logan Kelsch
Food Type Neural Network
'''
#IMPORT LIBRARIES AND FUNCTIONS
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

#load data from csv
data = pd.read_csv('FoodTypeDataset.csv',header=None)

#confirming X and Y features post training
Xfeatures = data.columns[:-1]
Yfeatures = data.columns[-1]
print("TESTED FEATURES: ")
print(Xfeatures)
print("TESTING FOR: ")
print(Yfeatures)

#sample count confirmation
print("OCCURANCES IN RAW DATA FOR ", Yfeatures, ": ", sep='')
unique, counts = np.unique(data.iloc[:, -1].values, return_counts=True)
print(dict(zip(unique,counts)))

#target class counts for developing class weights
last_column = data.iloc[:, -1].values
unique, counts = np.unique(last_column, return_counts=True)
class_counts = dict(zip(unique, counts))
#creating variable for use in .compile for model
classWeights = counts

# Separate features and target into X and y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#Encoding target
labelencoder = LabelBinarizer()
y = labelencoder.fit_transform(y)


#split data for training and examination
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#learning rate function for training optimization
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',#watch this value
    factor=0.90, #reduce to this percent when
    patience=32, #value hasn't improved in this long
    min_lr=1e-6 #and stop here
)

#different optimizers I used during training
opt1 = SGD(learning_rate=0.01)
opt2  = tf.keras.optimizers.Adam(clipnorm=0.7)
#commented out opt3 - contains deleted function
#opt3 = SGD(learning_rate=lr_schedule)
#this was the final optimizer I used.
opt4 = SGD(learning_rate=0.005, momentum=0.98)

#early stopping function to cut training short if the TRAINING recall stops improving
#this was decided as recall was the slowest to improve overall, and therefore
#was a great indicator of knowing when training is no longer productive
early_stopping = EarlyStopping(monitor='recall', patience=128, mode='max', restore_best_weights=True)

#function to construct model in one go
def build_model():
    '''
        maximized normalization in this model to combat agressive
        overfitting, as there are many features and classes, 
        and less samples to create strong model understandings
        used this approach as we were looking strictly for best validation
        performance output
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        #20 classes in output layer
        tf.keras.layers.Dense(20, activation='softmax')
    ])
    #custom metrics I wanted for verbose output
    met = ['precision','recall','accuracy']
    #compile all
    model.compile(optimizer=opt4,
                  loss='categorical_crossentropy'
                  ,metrics=met)
    return model

#considering the LR function and earlystopping
#I set epochs very high as I knew it would not get this far
epochs = 2000

#build and fit model
model = build_model()
#record history
history = model.fit(X_train, y_train, epochs=epochs,\
                    shuffle=True, verbose=1, validation_data=(X_test, y_test), \
                        callbacks=[early_stopping], batch_size=20, \
                            class_weight=classWeights)

#visualize model performance using matpltlib

#define for proper formatting
epochs = range(1, len(history.history['loss']) + 1)
#loss output
plt.figure(figsize=(12, 6))
plt.plot(epochs, history.history['loss'], 'y', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
#accuracy output
plt.plot(epochs, history.history['accuracy'], 'y', label='Training acc')
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
#precision output
plt.plot(epochs, history.history['precision'], 'y', label='Training Precision')
plt.plot(epochs, history.history['val_precision'], 'r', label='Validation Precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
#recall output
plt.plot(epochs, history.history['recall'], 'y', label='Training Recall')
plt.plot(epochs, history.history['val_recall'], 'r', label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

#predicting the test set results
y_pred = np.argmax(model.predict(X_test), axis=1) #no one-hot encoding needed for this dataset

#confusion matrix creation
cm = confusion_matrix(y_true, y_pred)

#confusion matrix output
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=range(20), yticklabels=range(20))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for 5-Class Classification')
plt.show()

#file was originally built and testing in a .ipynb
#for the seperated and primary use of saving the model
