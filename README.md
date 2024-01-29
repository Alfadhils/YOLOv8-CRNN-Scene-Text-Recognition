# YOLOv8-CRNN Scene Text Recognition

This repository is dedicated to implementing Deep Learning-based Scene Text Recognition models, utilizing a two-step approach involving Text Detection (TD) and Text Recognition (TR). The TD step employs YOLOv8, while the TR step utilizes a Convolutional Recurrent Neural Network (CRNN). The dataset used for training and evaluation can be accessed through this [Kaggle link](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset).

## Directory Structure

- The `runs/detect` directory stores training and validation information for YOLOv8 models.
- The `runs/crnn_train` contains result plots for CRNN models.

## Scripts and Notebooks

All scripts and notebooks are located under the `src/` directory:

1. **yolov8_datagen.py**

   - This Python script (`yolov8_datagen.py`) reformats the dataset into the YOLOv8 training format for TD.

2. **train_val_workflow.ipynb**

   - The notebook script (`yolov8_workflow.ipynb`) provides a step-by-step guide on custom training and evaluating YOLOv8 models using the data generation script (`yolov8_datagen.py`). It covers the entire workflow from data preparation to model training.

3. **result_analysis.ipynb**

   - The notebook script (`yolov8_results.ipynb`) focuses on results analysis and metric measurements for YOLOv8 trainings. This script aids in evaluating the performance of YOLOv8 models trained on datasets with varying text instance density.

4. **crnn_datagen.py**

   - The Python script (`crnn_datagen.py`) is used to crop text instances from images and create the TR dataset.

## Getting Started

To begin custom training and evaluation, follow the steps outlined in the `yolov8_workflow.ipynb` notebook. Additionally, refer to the `yolov8_results.ipynb` notebook for detailed metric measurements and analysis.

## Dependencies

Ensure all necessary dependencies are installed by running:

```bash
pip install -r requirements.txt
```

Feel free to customize and adapt the code to fit your specific requirements. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Contact

For any inquiries or feedback, please contact Fadhil Umar at [[fadhilumaraf.9a@gmail.com](mailto:fadhilumaraf.9a@gmail.com)].
