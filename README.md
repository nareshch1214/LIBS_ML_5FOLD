# LIBS_ML_5FOLD
This repository contains a MATLAB-based workflow for the classification of LIBS (Laser-Induced Breakdown Spectroscopy) spectral data. The process involves Principal Component Analysis (PCA) for dimensionality reduction, followed by the evaluation of multiple machine learning models using cross-validation.
Project Overview
The core objective of this project is to classify different samples based on their LIBS spectra. The workflow is divided into three main stages, each corresponding to a MATLAB script:
PCA and Visualization: The first script processes the raw spectral data, normalizes it, and applies PCA. It then generates 3D plots to visualize the separation of the sample groups and a bar plot of explained variance.
Machine Learning Evaluation: The second script uses the PCA-transformed data to train and evaluate seven different classification models using 5-fold cross-validation. It calculates and saves key performance metrics (Accuracy, Precision, Recall, F1-Score, MSE) to an Excel file and a bar chart.
Confusion Matrix Generation: The final script creates cross-validated confusion matrices for each classifier, which are essential for understanding the model's performance on a class-by-class basis.
Prerequisites
MATLAB: This project requires MATLAB to run. It was developed and tested using MATLAB R2023a but should be compatible with recent versions that include the Statistics and Machine Learning Toolbox.
Data File: An Excel spreadsheet (.xlsx) containing your LIBS data. The file must be structured as follows:
Column 1: Wavelength values.
Column 2 onwards: Intensity values for each spectrum.
The code assumes a total of 60 spectra, organized into four sample groups (U0, U5, G0, G5), with 15 spectra per group.
How to Run the Code
Follow these steps in the specified order.
Place the Files: Ensure all three MATLAB scripts (first_file.m, second_file.m, and third_file.m) and your LIBS data Excel file are in the same folder.
Open MATLAB: Launch MATLAB and navigate to the folder containing your files.
Run the First Script: Execute the first script by typing first_file in the MATLAB Command Window and pressing Enter.


first_file: PCACODE.m


A file dialog will appear. Select your LIBS data Excel file.
This script will perform PCA and generate two figures: a 3D PCA plot and an explained variance bar chart.
Run the Second Script: After the first script completes, run the second one by typing second_file in the Command Window.


second_file: PCACODE_5Fold.m


This script will train the machine learning models, print the cross-validation results to the Command Window, and create a bar chart of the mean accuracies. It will also save the full results to a file named Classifier_Performance_Results.xlsx.
Run the Third Script: Finally, run the third script by typing third_file in the Command Window.


third_file: PCACODE_5Fold_CONTDD.m


This script will generate a comprehensive figure with all the cross-validated confusion matrices. It will also save individual, high-resolution confusion matrix figures for each classifier in the same folder.
