# YOLOv8-CRNN Scene Text Recognition

This repository is dedicated to implementing Deep Learning-based Scene Text Recognition models, utilizing a two-step approach involving Text Detection (TD) and Text Recognition (TR). The TD step employs YOLOv8, while the TR step utilizes a Convolutional Recurrent Neural Network (CRNN). The dataset used for training and evaluation are provided by TextOCR with ~1M high quality word annotations on TextVQA images. The dataset can be accessed through this [Kaggle link](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset).

Download or use the Kaggle API to download and extract the dataset. initialize the dataset by storing the extracted Kaggle dataset in `datasets` directory. The directory should look something like this.

├── datasets  
│   └── archive  
│       ├── train_val_images  
│       │   └── train_images  
│       │       ├── img1.jpg  
│       │       ├── img2.jpg  
│       │       └── ...   
│       ├── annot.csv  
│       ├── annot.parquet  
│       ├── img.csv  
│       ├── img.parquet  
│       └── TextOCR_0.1_train.json  
├── demo  
└── ...  


## Scripts and Notebooks

All scripts and notebooks are located under the `src/` directory:

1. **yolov8_datagen.py**

   - This Python script (`yolov8_datagen.py`) reformats the dataset into the YOLOv8 training format for TD.

2. **yolov8_workflow.ipynb**

   - The notebook script (`yolov8_workflow.ipynb`) provides a step-by-step guide on custom training and evaluating YOLOv8 models using the data generation script (`yolov8_datagen.py`). It covers the entire workflow from data preparation to model training.

3. **crnn_datagen.py**

   - This Python script (`crnn_datagen.py`) is responsible for cropping text instances from images and creating the TR dataset.

4. **crnn_dataset.py**

   - The Python script (`crnn_dataset.py`) manages the dataset for the CRNN model, including loading and preprocessing.

5. **crnn_decoder.py**

   - The Python script (`crnn_decoder.py`) contains the decoder implementation for the CRNN model.

6. **crnn_model.py**

   - The Python script (`crnn_model.py`) defines the architecture of the CRNN model for text recognition.

7. **crnn_predict.py**

   - This Python script (`crnn_predict.py`) is used for predicting text instances using the trained CRNN model.

8. **crnn_train.py**

   - The Python script (`crnn_train.py`) is responsible for training the CRNN model on the prepared dataset.

9. **crnn_evaluate.py**

   - The Python script (`crnn_evaluate.py`) evaluates the performance of the trained CRNN model on a validation set.

10. **predict.py**

    - The Python script (`predict.py`) performs inference of the full STR workflow using YOLOv8 detector and CRNN recognizer.

## Getting Started

1. To begin using the pretrained YOLO Text Detector, you can use Ultralytics YOLO API through the CLI.
```bash 
yolo detect predict model=checkpoints/yolov8_5k.pt source=demo/TD.jpg
```

2. To begin using the pretrained CRNN Text Recognizer, you can use the `crnn_predict.py` script throught eh CLI.
```bash 
python src/crnn_predict.py --cp_path checkpoints/crnn_s100k.pt --source demo/TR_Harris.png
```
python src/crnn_predict.py --cp_path checkpoints/crnn_s100k.pt --source demo/TR_Harris.png

3. To begin using the full STR workflow of both YOLOv8 and CRNN, you can use the `predict.py` script trorugh the CLI.
  - Image
```bash 
python src/predict.py --detector checkpoints/yolov8_5k.pt --recognizer checkpoints/crnn_s100k.pt --source demo/TD.jpg
```
  - Video
```bash 
python src/predict.py --detector checkpoints/yolov8_5k.pt --recognizer checkpoints/crnn_s100k.pt --source demo/street.mp4
```

## Dependencies

Ensure all necessary dependencies are installed by running:

```bash
pip install -r requirements.txt
```

Feel free to customize and adapt the code to fit your specific requirements. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

## References
This project are heavliy inspired by Ultralytics and CRNN-Pytorch Github Repo:
1. Ultralytics Documentation Page [Ultralytics](https://github.com/ultralytics/ultralytics)
2. CRNN Implementation on Pytorch Github [CRNN-Pytorch](https://github.com/GitYCC/crnn-pytorch)
3. Original CRNN Research Paper [CRNN](https://arxiv.org/abs/1507.05717)
4. TextOCR - Text Extraction from Images Dataset, Kaggle Link [TextOCR](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset/data)

## Contact

For any inquiries or feedback, please contact Fadhil Umar at [[fadhilumaraf.9a@gmail.com](mailto:fadhilumaraf.9a@gmail.com)].
