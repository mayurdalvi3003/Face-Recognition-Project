# **Face Recognition Project**
This repository contains a project on face recognition using machine learning techniques. The implementation is based on the Labeled Faces in the Wild (LFW) dataset. Below is an outline of the preprocessing steps and methodologies used in the project.

# Steps and Descriptions
## 1. Open Google Colab and upload the `.ipynb` file over there and follow the following procedure
## 2. Loading the Dataset
- Dataset: Labeled Faces in the Wild (LFW)
- Code Snippet:
```python
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
```
- The dataset includes images of faces labeled with names of people, with a minimum of 60 faces per person.
### Classes Identified:
- Ariel Sharon
- Colin Powell
- Donald Rumsfeld
- George W Bush
- Gerhard Schroeder
- Hugo Chavez
- Junichiro Koizumi
- Tony Blair
- Shape: (1348, 62, 47) - 1348 images, each of 62x47 resolution.

## 3. Visualizing the Dataset
- Plots sample images from the dataset for visualization.
- Code Snippet:
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 5, figsize=(16, 9))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
```
- Visualizes 15 images of faces from the dataset.
  
## 4. Setting Up the Classification Pipeline
### Components:
- Principal Component Analysis (PCA): Dimensionality reduction with 150 components.
- Support Vector Machine (SVM): For classification with an RBF kernel.

### Pipeline Setup:
```python
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
```

## 5. Data Splitting
- The dataset is split into training and testing sets using train_test_split.
- Code Snippet:
```python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target, random_state=42)
```

## 6. Hyperparameter Tuning
- Uses GridSearchCV to optimize the C and gamma parameters of the SVM.
- Code Snippet:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, Ytrain)
print(grid.best_params_)
```
### Best Parameters Found:
- C: 5
- gamma: 0.001

## 7.Model Training and Prediction
The best model from GridSearchCV is trained on the training data.
Code Snippet:

```python
model = grid.best_estimator_
yfit = model.predict(Xtest)
```
## 8.Evaluation
- Evaluates the model using a classification report, confusion matrix, and accuracy metrics.
### Classification Report:
- Average accuracy: 85%
- Weighted precision: 86%
- Confusion Matrix Visualization:
```python
import seaborn as sns
mat = confusion_matrix(Ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
```
- Metrices
```python
from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(Ytest, yfit)
precision = precision_score(Ytest, yfit, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
```
## 9. Visualization of Results
- Displays target and predicted labels for test set images.
- Code Snippet:
```python
fig, axes = plt.subplots(5, 5, figsize=(16, 9))
for i, ax in enumerate(axes.flat):
    if i < len(Xtest):
        ax.imshow(Xtest[i].reshape(faces.images[0].shape), cmap='bone')
        target_name = faces.target_names[Ytest[i]]
        predicted_name = faces.target_names[yfit[i]]
        title_color = 'green' if target_name == predicted_name else 'red'

        ax.set_title(f"Target: {target_name}\nPredicted: {predicted_name}", color=title_color)
        ax.axis('off')
```
