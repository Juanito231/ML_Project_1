# Heart Disease Prediction
This project is a part of the [EPFL Machine Learning Course CS-433](https://github.com/epfml/ML_course).

# Project Description
The goal of this project is to use data from the Behavioral Risk Factor Surveillance System in order to predict whether the health situation of an individual can lead to a MICHD (coronary heart disease).

# Contributors
TODO

# Files Description
* `implementations.py`: the 6 mandatory functions
* `run.py`: file to reproduce our best result on AICrowd
* `helpers_given.py`: functions given by the teachers
* `helpers.py`: utility functions useful for the 6 mandatory methods and the prediction
* `preprocess.py`: utility functions useful for the preprocessing of the data
* `cross_validation.py`: utility functions useful for performing cross validation

# Methods used
- Mean Squared Error with Gradient Descent
- Mean Squared Error with Stochastic Gradient Descent
- Least Squares
- Ridge Regression
- Logistic Regression (with Gradient Descent)
- Regularized Logistic Regression (with Gradient Descent)

# How to get started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. The raw data is available [here](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/dataset_files). Download the data in csv format and store it in a folder named "dataset".
3. Make sure to have have all the libraries installed. TODO: include a list of all libraries needed

# Reproduce our best model
To reproduce our best result on AICrowd (accuracy=TODO, F1 score=TODO), change the path name to where your dataset folder is located. Then run the file `run.py`.
The results will be saved as a csv file named TODO.
