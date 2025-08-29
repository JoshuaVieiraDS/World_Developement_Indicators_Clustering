# Clustering Analysis on Interpolated Dataset 

## Overview
This Jupyter Notebook performs clustering on an interpolated dataset using K-Means (regular and K++) and Gaussian Mixture Models (GMM). Dimensionality reduction techniques like PCA and t-SNE are applied for visualization. Cluster quality is evaluated using silhouette scores, and top features contributing to principal components are identified.

## Files
- `Clustering_Notebook.ipynb`: Jupyter Notebook performing preprocessing, clustering, PCA/t-SNE visualization, and cluster evaluation.
- `Report.pdf`: A report with the results and evaluations.

## Requirements
```python
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
