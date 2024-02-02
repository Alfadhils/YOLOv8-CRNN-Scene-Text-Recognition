import torch
from torch import nn
from tqdm import tqdm

from crnn_dataset import get_split, TRDataset
from crnn_model import CRNN
from crnn_decoder import ctc_decode

import argparse

# Example Usage
# python src/crnn_evaluate.py --cp_path checkpoints/crnn_s100k.pt

def get_input_args():
    """
    Get command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Generator")
    parser.add_argument("--cp_path", type=str, default='checkpoints/crnn_s100k.pt', help="Configuration checkpoint path.")
    return parser.parse_args()

def evaluate(crnn, dataloader, criterion, max_iter=None):
    """
    Evaluate a CRNN model on a given dataloader.

    Parameters:
    - crnn (torch.nn.Module): The CRNN model to be evaluated.
    - dataloader (torch.utils.data.DataLoader): DataLoader containing evaluation data.
    - criterion (torch.nn.Module): The loss criterion for evaluation.
    - max_iter (int): Maximum number of iterations to evaluate. If None, evaluate on the entire dataloader.

    Returns:
    - dict: Dictionary containing evaluation metrics including loss, accuracy, and wrong_cases.
    """
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation

def main():
    args = get_input_args()
    
    reload_checkpoint = args.cp_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if reload_checkpoint:
        config = torch.load(reload_checkpoint, map_location=device)
    else :
        config = {
            'state_dict' : None,
            'img_height' : 32,
            'img_width' : 100,
            'batch_size' : 64,
            'root_dir' : "datasets/TR_100k",
            'labels' : "labels.csv",
            'splits' : [0.98,0.01,0.01],
            'map_to_seq' : 64,
            'rnn_hidden' : 256
        }
    
    val_loader = get_split(root_dir=config['root_dir'],
                           labels=config['labels'],
                           set='val',
                           img_width=config['img_width'],
                           img_height=config['img_height'],
                           batch_size=config['batch_size'],
                           splits=config['splits'])
    
    num_class = len(TRDataset.LABEL2CHAR) + 1
    
    crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                map_to_seq=config['map_to_seq'],
                rnn_hidden=config['rnn_hidden'])

    if config['state_dict']:
        crnn.load_state_dict(config['state_dict'])

    crnn.to(device)
    
    criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
    
    print(f"Evaluating [{config['root_dir']}] using {device}")
    evaluation = evaluate(crnn, val_loader, criterion)
    print(f"Val Loss : {evaluation['loss']:.4f}, Val Acc: {evaluation['acc']}")
    
if __name__ == "__main__":
    main()   
