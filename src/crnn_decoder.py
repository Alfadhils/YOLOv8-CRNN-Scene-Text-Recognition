import numpy as np

def _reconstruct(labels, blank=0):
    """
    Merge consecutive identical labels and remove blank labels.

    Parameters:
    - labels (list): List of integer labels.
    - blank (int): Integer representing the blank label.

    Returns:
    - list: List of reconstructed labels without consecutive duplicates and blanks.
    """
    new_labels = []
    previous = None
    for label in labels:
        if label != previous:
            new_labels.append(label)
            previous = label
    new_labels = [l for l in new_labels if l != blank]
    return new_labels

def greedy_decode(emission_log_prob, blank=0):
    """
    Perform greedy decoding on emission probabilities.

    Parameters:
    - emission_log_prob (numpy.ndarray): 2D array of emission log probabilities.
    - blank (int): Integer representing the blank label.

    Returns:
    - list: List of decoded labels after greedy decoding.
    """
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels

def ctc_decode(log_probs, label2char=None, blank=0):
    """
    Perform CTC decoding on log probabilities.

    Parameters:
    - log_probs (numpy.ndarray): 3D array of log probabilities.
    - label2char (dict): Optional dictionary mapping labels to characters.
    - blank (int): Integer representing the blank label.

    Returns:
    - list: List of decoded sequences after CTC decoding.
    """
    # Transpose log_probs for convenient iteration over time steps
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    
    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = greedy_decode(emission_log_prob, blank=blank)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list
