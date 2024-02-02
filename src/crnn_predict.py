import torch
from PIL import Image
import time
import argparse
from crnn_dataset import TRDataset, preprocess
from crnn_model import CRNN
from crnn_decoder import ctc_decode

def get_input_args():
    """
    Get command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CRNN Text Recognition Prediction")
    parser.add_argument("--cp_path", type=str, default=None, help="Configuration checkpoint path.")
    parser.add_argument("--source", type=str, default=None, help="Prediction source path.")
    return parser.parse_args()

def predict(crnn, data_loader, label2char=None):
    """
    Perform text prediction using the CRNN model.

    Args:
        crnn (CRNN): The CRNN model.
        data_loader (iterable): DataLoader containing input images.
        label2char (dict): Mapping from label index to character.

    Returns:
        list: List of predicted texts.
    """
    crnn.eval()
    with torch.no_grad():
        for data in data_loader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            images = data.to(device)
            
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            
            preds = ctc_decode(log_probs, label2char=label2char)
            texts = [''.join(pred) for pred in preds]
    
    return texts

def main():
    args = get_input_args()
    start = time.perf_counter()
    img = Image.open(args.source)
    
    reload_checkpoint = args.cp_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if reload_checkpoint:
        config = torch.load(reload_checkpoint, map_location=device)
    else:
        print("No checkpoint loaded, using default configuration.")
        config = {
            'state_dict': None,
            'img_height': 32,
            'img_width': 100,
            'batch_size': 64,
            'root_dir': "datasets/TR_100k",
            'labels': "labels.csv",
            'splits': [0.98, 0.01, 0.01],
            'map_to_seq': 64,
            'rnn_hidden': 256
        }
    
    # Preprocess input image
    img = preprocess(img, config['img_height'], config['img_width'])
    img = img.unsqueeze(0)
    
    num_class = len(TRDataset.LABEL2CHAR) + 1
    
    crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                map_to_seq=config['map_to_seq'],
                rnn_hidden=config['rnn_hidden'])

    if config['state_dict']:
        crnn.load_state_dict(config['state_dict'])

    crnn.to(device)
    
    results = predict(crnn, [img], TRDataset.LABEL2CHAR)
    end = time.perf_counter()
    
    print(f"Prediction in {end - start} seconds")
    print(f"Results: {results[0]}")
    
if __name__ == "__main__":
    main()
