from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import time

from crnn_dataset import TRDataset
from crnn_model import CRNN
from crnn_decoder import ctc_decode
from crnn_predict import predict

import argparse

def get_input_args():
    """
    Get command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Text Recognition using YOLOv8 and CRNN")
    parser.add_argument("--detector", type=str, default=None, help="YOLOv8 weights path.")
    parser.add_argument("--recognizer", type=str, default=None, help="CRNN checkpoint configuration path.")
    parser.add_argument("--source", type=str, default=None, help="Prediction source path.")
    return parser.parse_args()

def text_recognition(img_path, yolo_weight, crnn_config):
    """
    Perform text recognition using YOLOv8 for text detection and CRNN for text recognition.

    Args:
        img_path (str): Path to the input image.
        yolo_weight (str): Path to YOLOv8 weights.
        crnn_config (str): Path to CRNN checkpoint configuration.

    Returns:
        Image: Resultant image with annotated text.
    """
    start = time.time()
    model = YOLO(yolo_weight)
    
    if type(img_path) is str:
        org_img = Image.open(img_path)
    else:
        org_img = img_path

    results = model(org_img, conf=0.4, verbose=False)
    
    cropped_texts = extract_texts(results, org_img)
            
    config = torch.load(crnn_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if cropped_texts:
        detect_dataset = TRDataset(images=cropped_texts, img_height=config['img_height'], img_width=config['img_width'])
        detect_loader = torch.utils.data.DataLoader(detect_dataset, batch_size=64, shuffle=False)

        num_class = len(TRDataset.LABEL2CHAR) + 1

        crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                    map_to_seq=config['map_to_seq'],
                    rnn_hidden=config['rnn_hidden'])

        if config['state_dict']:
            crnn.load_state_dict(config['state_dict'])

        crnn.to(device)

        texts = predict(crnn, detect_loader, label2char=TRDataset.LABEL2CHAR)
    else:
        print('No text detected')
        return org_img

    result_image = annotator(org_img, results, texts)
            
    end = time.time()
    print(f"Total time : {end - start}, Detected {len(cropped_texts)} texts")
    return result_image

def extract_texts(results, img):
    """
    Extract cropped images containing detected texts.

    Args:
        results: YOLOv8 detection results.
        img (PIL.Image): Original input image.

    Returns:
        list: List of cropped images containing detected texts.
    """
    cropped_texts = []
    for r in results:
        for text in r.boxes.xywh:
            x, y, w, h = text.cpu().numpy()
            cropped_img = img.crop((x - w/2, y - w/2, x + w/2, y + h/2))
            cropped_texts.append(cropped_img)
            
    return cropped_texts

def annotator(img, results, texts):
    """
    Annotate the original image with bounding boxes and recognized texts.

    Args:
        img (PIL.Image): Original input image.
        results: YOLOv8 detection results.
        texts (list): List of recognized texts.

    Returns:
        PIL.Image: Annotated image.
    """
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arialbd.ttf", size=16)

    for r in results:
        for (x, y, w, h), text in zip(r.boxes.xywh.cpu().numpy(), texts):
            draw.rectangle([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)], outline="red", width=2)
            draw.text((int(x - w/2), int(y - h/2 - 18)), text, fill="red", font=font)
    
    return img
    

def main():
    args = get_input_args()
    
    img_path = args.source
    detector = args.detector
    recognizer = args.recognizer
    
    result = text_recognition(img_path, detector, recognizer)
    result.save('result.png')
    
    result_array = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    cv2.imshow('Result Image', result_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
