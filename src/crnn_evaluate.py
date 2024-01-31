import torch
from torch import nn
from src.crnn_decoder import ctc_decode
from tqdm import tqdm

from crnn_dataset import get_split, TRDataset
from crnn_model import CRNN

import argparse

def get_input_args():
    """
    Get command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Generator")
    parser.add_argument("--source_dir", type=str, default='../datasets/archive', help="Source directory path")
    parser.add_argument("--dest_dir", type=str, default='../datasets/cropped_text', help="Destination directory path")
    parser.add_argument("--total_images", type=int, default=10000, help="Total number of images")
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
    
    img_height, img_width = 32,100
    
    val_loader = get_split(root_dir="datasets/TR_100k",
                           labels="labels.csv",
                           set='val',
                           img_width=img_width,
                           img_height=img_height,
                           batch_size=64,
                           splits=[0.98,0.01,0.01])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_class = len(TRDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq=64,
                rnn_hidden=256)
    
    reload_checkpoint = 'checkpoints/crnn_100k.pt'
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)
    
    criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=0.0005)
    
    evaluation = evaluate(crnn, val_loader, criterion)
    print(f'Evaluating [{reload_checkpoint}] using {device}')
    print(f"Val Loss : {evaluation['loss']:.4f}, Val Acc: {evaluation['acc']}")
    
if __name__ == "__main__":
    main()   
