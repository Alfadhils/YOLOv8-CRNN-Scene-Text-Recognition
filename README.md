# YOLOv8 Scene Text Detection

This repository focuses on training YOLOv8 models for scene text detection using custom datasets with varying text instance density. The project aims to assess the impact of class imbalance on text detection performance by utilizing three different datasets.

## Directory Structure

- The `runs/detect` directory stores training and validation information for each dataset separately.

## Scripts and Notebooks

1. **yolov8datagen.py**

   - This Python script (`yolov8datagen.py`) is used to reformat the Kaggle dataset into the YOLOv8 training format.
2. **train_val_workflow.ipynb**

   - The notebook script (`train_val_workflow.ipynb`) provides a step-by-step guide on how to custom train and evaluate YOLOv8 models using the data generation script (`yolov8datagen.py`). It covers the entire workflow from data preparation to model training.
3. **result_analysis.ipynb**

   - The notebook script (`result_analysis.ipynb`) focuses on different metric measurements for each dataset provided and includes in-depth analysis. This script helps in evaluating the performance of YOLOv8 models trained on datasets with varying text instance density.

## Dataset Information

- Three different datasets with varying text instance density are used to analyze the effect of class imbalance on text detection.

## Getting Started

To get started with custom training and evaluation, follow the steps outlined in the `train_val_workflow.ipynb` notebook. Additionally, refer to the `result_analysis.ipynb` notebook for detailed metric measurements and analysis.

## Dependencies

Make sure to install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

Feel free to customize and adapt the code to fit your specific requirements. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Contact

For any inquiries or feedback, please contact Fadhil Umar at [[fadhilumaraf.9a@gmail.com](mailto:fadhilumaraf.9a@gmail.com)].
