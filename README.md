# Tensor-Transformer-for-HSIC

An example program for Tensor Transformer (TT) is provided. Our publication can be found at: [https://doi.org/10.1016/j.patcog.2025.111470](https://doi.org/10.1016/j.patcog.2025.111470)

## Project Structure

The files and directories are described as follows:

- **Dataset/**: Contains the raw data and label information of the LongKou (LK) dataset. LK data can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1sZqjIGaAYLa4LMSaKsoqQg?pwd=8216).
- **config.py**: Global configuration file. No modification is required if only performing full-image prediction.
- **TensorTransformerFramework.py**: The main framework code of the TT network, providing a model definition example for reference and further modification.
- **LibTNN.py**: Contains custom neural network layer modules required for the TT model.
- **img_predict.py**: Model prediction script for loading trained models and performing full-image prediction.
- **evaluation.py**: Evaluation script for generating visualizations of prediction results and computing relevant evaluation metrics.

## Pre-trained Models

Additionally, the project provides pre-trained model parameters and their prediction results:

- **TensorTransformerHSI_unoverlapping_LK.pth**: Model parameters trained on the LK dataset without overlapping sampling.
- **TensorTransformerHSI_unoverlapping_LK_predict_result_filter.mat**: Prediction results of the corresponding model on the LK dataset.