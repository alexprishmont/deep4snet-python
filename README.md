# Deep4SNet Audio Authenticity Analysis
This repository contains a Python implementation of the Deep4SNet model, which is used for analyzing audio files (.wav format) to determine their authenticity. The implementation focuses on identifying whether an audio file is original or has been artificially generated (deepfake).

# Getting Started
## Prerequisites
Before running this application, ensure you have Python installed on your system.

- Python 3.10


## Installation
Clone this repository to your local machine:


```bash 
git clone https://github.com/alexprishmont/deep4snet-python.git && cd deep4snet-python
```

Install all the dependencies:
```bash
pip install -r requirements.txt
```
# Using the Pre-trained Models
The pre-trained models used in this project are courtesy of the original Deep4SNet repository. Make sure to download the required model files from Deep4SNet Repository and place them in the appropriate directory in your local clone of this repository.

Running the Application
To run the application, use the following command:

```bash
python3 main.py path/to/your/audio.wav
```
Replace path/to/your/audio.wav with the actual path to the audio file you want to analyze.

# How It Works
The application processes the provided audio file, converts it into a spectrogram, and then utilizes the Deep4SNet model to predict whether the audio is original or a deepfake. The result is displayed in the console.

# Acknowledgments
This project utilizes the Deep4SNet model
developed by Yohanna Rodriguez. 
The pre-trained models and additional resources can be found in the [Deep4SNet GitHub repository](https://github.com/yohannarodriguez/Deep4SNet).