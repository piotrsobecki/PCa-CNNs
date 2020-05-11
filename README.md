# Prostate lesion classification using Deep Convolutional Neural Networks

We propose three novel VGG based architectures for prostate lesion classification (see [Architectures module](src/cnn/architectures)):
- CNN_VGG_SIMPLE with three subnetworks for mpMRI modalities 
- CNN_VGG_MODALITIES with mixture of experts architecture (each modality is capable of making predictions)
- CNN_VGG_PIRADS with a-priori knowledge embedding in mixture of experts architecture 

This project uses data from [ProstateX1 competition](https://spiechallenges.cloudapp.net/competitions/6).

Directory [data/ProstateX](data/ProstateX) contains trail dataset, a preselected cases from ProstateX1 Challenge. In order to perform analysis on bigger dataset - download the data from original sources and extract them to directory following the trail directory as an example.


## Installation

Install Anaconda 4.3.1 (with Python 3.6.0) - https://repo.continuum.io/archive/
Execute setup.bat (or copy content to terminal on unix)
Note that requirements.txt may not be complete - if so, please add missing requirements to the text file and make pull request

## Running

After installation, to run the tests:
- Run [augment_data.py](src/augment_data.py) to augment and save locally the data
- Modify [constants.py](src/constants.py) to match your experiment parameters
- Run [test.py](src/test.py) to train the model using CV and verbose logging
