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

## Workflow
The project workflow is straightforward: Given an image, text detection and recognition are performed through YOLOv8 and CRNN models, respectively. The process involves the detection and extraction of texts using YOLOv8, storing the resulting texts as a collection of cropped text images. These cropped images serve as input for the CRNN model, which recognizes all the text within them. The final results are then plotted on the original image. The illustration of the workflow is presented below:
![workflow](images\workflow.png)


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

The demo video is a portion of streetview video from Walking Around youtube channel. The full video can be accessed [here](https://www.youtube.com/watch?v=_2oJYWBFdMg) 

## Results 
1. YOLOv8 Small Model for Text Detection
   - **Training Details:**
      - Model: YOLOv8 Small
      - Fine-tuned on 5,000 images from the TextOCR dataset
      - Training Epochs: 20

   - **Losses:**
      - Train Box Loss: 1.305
      - Validation Box Loss: 1.2908

   - **Performance Metrics:**
      - Mean Average Precision (mAP50): 67.559%

2. CRNN Pretrained Model for Text Recognition

   - **Training Details:**
      - Model: CRNN
      - Pretrained on synth90k dataset [link](https://github.com/GitYCC/crnn-pytorch)
      - Fine-tuned on 100,000 cropped text images from the TextOCR dataset
      - Training Epochs: 5
      - CTC Decoder: Greedy algorithm for faster inference time

   - **Losses:**
      - Train CTC Loss: 5.948
      - Validation CTC Loss: 4.664

   - **Accuracy:**
      - Validation Accuracy: 58%

for more information about training, evaluation, and dataset generation refer to source code.

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

## Citations
```
@software{Jocher_Ultralytics_YOLO_2023,
      author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
      license = {AGPL-3.0},
      month = jan,
      title = {{Ultralytics YOLO}},
      url = {https://github.com/ultralytics/ultralytics},
      version = {8.0.0},
      year = {2023}
}

@inproceedings{singh2021textocr,
      title={{TextOCR}: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text},
      author={Singh, Amanpreet and Pang, Guan and Toh, Mandy and Huang, Jing and Galuba, Wojciech and Hassner, Tal},
      journal={The Conference on Computer Vision and Pattern Recognition},
      year={2021}
}

@misc{shi2015endtoend,
      title={An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition}, 
      author={Baoguang Shi and Xiang Bai and Cong Yao},
      year={2015},
      eprint={1507.05717},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Contact

For any inquiries or feedback, please contact Fadhil Umar at [[fadhilumaraf.9a@gmail.com](mailto:fadhilumaraf.9a@gmail.com)].
