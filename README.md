# **Face Recognition Project**
This repository contains a project on face recognition using machine learning techniques. The implementation is based on the Labeled Faces in the Wild (LFW) dataset. Below is an outline of the preprocessing steps and methodologies used in the project.

# Steps and Descriptions
## 1. Loading the Dataset
- Dataset: Labeled Faces in the Wild (LFW)
- Code Snippet:
```python
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
```
