#Deep Learning via ANN
# Importing The Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns  # <-- For enhanced visualization

# Importing dataset

dataset = pd.read_csv("C:/Users/kumar_/Desktop/CSE/PYTHON/Project/Dataset_master.csv")
# Drop rows with missing values
dataset.dropna(inplace=True)
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

print(X)

#Encoding categorical data

# Gender column : Label Encoding

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[:,2] = label_encoder.fit_transform(X[:,2])

print(X)

# Geography column:One hot Encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

print(X)

# Splitting daaset into Training & Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN
# Initialization

ann = tf.keras.models.Sequential()

# Adding input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

# Adding second hidden layer
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

# Adding output layer
ann.add(tf.keras.layers.Dense(units=1,  activation='sigmoid'))

# ANN Training
# Training on training dataset
# 1. Compile the model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 2. Train the model
history = ann.fit(X_train, y_train, batch_size=32, epochs=120)

# Compiling ANN
# Compile the model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training on training dataset
ann.fit(X_train, y_train, batch_size=32, epochs=120)

# Predictions
# Single Prediction
print(ann.predict(sc.transform([[0.0, 1.0, 0.0, 501, 0, 32, 2, 0.0, 4, 1, 545501]])) > 0.5)


# Predection on testset
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))  # Print the accuracy

# Matplotlib Visualization (Added Section)

# Accuracy & Loss curves
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy', color='green')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss', color='red')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
# Visualization after model training
import matplotlib.pyplot as plt
import seaborn as sns  # For improved plot styling

sns.set(style="darkgrid")  # Apply dark grid style for better readability

# Plot training & validation accuracy values
plt.figure(figsize=(14, 6))

# Subplot for accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], color='green', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', linestyle='--', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Subplot for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color='red', label='Training Loss')
plt.plot(history.history['val_loss'], color='orange', linestyle='--', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
