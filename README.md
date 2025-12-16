# CSC_51073_CV_Project
Repository for the end of trimester project of a computer vision course. We aim to implement a real-time fitness exercise tracker for healthcare applications

Files:
 - main.py: pipeline file for processing our algorithm on a single video
 - display.py: functions for playing the video with landmarks and results
 - classifier.py: code for exercise classification
 - ExerciseClasses.py: implements the exercice classes parameters for the grading
 - grader.py: functions for grading the execution of an exercise
 - landmarks_extraction.py: functions for the landmarks extraction from raw videos
 - metric.py: mathematical functions and tools used in the grading code
 - Model_building_and_training.ipynb: Code for building and training the classification model

Folders:
 - models (gitignored): contains the parameters of the trained CNN and PCA
 - ref: contains the landmarks files for the reference execution of each exercise
 - data: contains a few test videos for running the pipeline, and the landmarks dataset which was used to train the models

To run the pipeline, please choose a video in the data file, edit the filename line in main.py file and run main.py

Link to the kaggle dataset: https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video
