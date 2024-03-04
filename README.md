MLMA Final Project:

A good starting point for this project is to read the `report.pdf` file. 
It contains a detailed explanation of the project, the methods used, and the results obtained.

-`scripts`: some scripts for plotting graphs

- `config.py`: Configuration settings for the project.
- `dataset.py`: Handles data loading and preprocessing.
- `dwt.py`: Discrete Wavelet Transform implementation.
- `extract_features.py`: Feature extraction from signals.
- `freq_feature.py`: Frequency domain feature extraction.
- `model.py`: Defines the machine learning models.
- `plot_loss.py`: Visualization of training loss.
- `random_forest.py`: Random forest algorithm implementation.
- `resnet.py`: Residual Network (ResNet) model implementation.
- `svm.py`: Support Vector Machine (SVM) algorithm implementation.
- `test_model.py`: Script to test the models' performance.
- `train_model.py`: Training the machine learning models.
- `utils.py`: Utility functions for the project.



Datasets and Feature Data link:
https://drive.google.com/drive/folders/1wCRhijtqMATvnAVqeoBDG5v_lqru2uRn?usp=sharing

The 'raw_data' folder contains datasets from the PhysioNet/Computing in Cardiology Challenge 2016. These are unprocessed raw heart sound audio files.
The all_dwt folder contains the feature data  and labels I've extracted from these raw heart sounds.

The files and labels in the 'all_dwt' folder are utilized for model training.


Feel free to download the original datasets from the official website and use the scripts in this repository to extract the features and train the models!


How to install the required packages:
```bash
pip install -r requirements.txt
```