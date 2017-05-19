# Blind Image Inpainting using Autoencoding Neural Networks

Patrik Barkman, Axel Demborg and Mårten Nilsson

## Instalation
* The scripts require among others ImageMagic, Python 3 and Tensorflow

* Clone this repository

* Run the make file in `data` to download the data set

## Data preparation

* Run `python proces_lfw_data.py` to down-sample and generate distorted samples

* Run `python data_to_npz.py` to generate numpy matrices of the data.

## Running

* Run `python train.py data/numpy_data.npz 1000 100 log board` to train on the recently generated data for 1000 epocs, batch size 100 and writing logs to tensorboard.

* Run `python eval.py data/numpy_data.npz 1000` to evaluate on the training data, 
