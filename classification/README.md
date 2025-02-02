# CS 4/591 Assignment 2: Basic Models for Classification

## Group Members
- Jyrus Cadman
- Robert McCourt
- Bethany Pena
- Gabriel Urbaitis

## Purposes

We are going to evaluate the performance of different classification models.

## Implementation Tasks

1. [ ] Write a Python class for each of the following classification methods:
  - [ ] Widrow-Hoff Learning (Robert);
  - [ ] Linear Support Vector Machine (SVM) (Bethany);
  - [ ] Logistic Regression (Jyrus).

  Each class should have a constructor which initializes the weights and bias, a function `forward` which computes the output of the ML model based on a given input, a function `fit` which trains the ML model based on the given set of training samples.

2. [ ] Read the data set library from [scikit-learn](https://scikit-learn.org/stable/api/sklearn.datasets.html). Choose at least two sets of data for binary classification and test the performance of the above 3 ML models. You need to use the same training and testing samples.
  - You may choose a large number for the training and testing data, and compare the prediction accuracies of the models.
  - You may also choose the same learning rate, to see which model spends the least time cost.

3. [ ] Write a Python class for the Weston-Watkins SVM (Gabriel).

4. [ ] Test the Weston-Watkins SVM using a scikit-learn data set for multiclass classification (Gabriel).


## How to use:

To test the binary classification models (widrow-hoff, linear SVM, logistic regression), run the files 'test_model_breast_cancer.py'
to test the models effectiveness on the breast cancer datasets and 'test_model_faces.py' to test the models effectiveness on the wine dataset.

To test the multiclass classification model (Weston-Watkins SVM), run the file 'weston_watkins.py'.