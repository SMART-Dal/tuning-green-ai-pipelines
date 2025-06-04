import torch.nn as nn


def layer_drop(layer_list, N, position):
    """
    Drops a specified number of layers from a language model's layer list, either from the top or bottom.
    
    Args:
        layer_list (nn.ModuleList): The list of layers to modify (e.g., model.bert.encoder.layer).
        N (int): Number of layers to drop.
        position (str): Where to drop layers from, either 'top' or 'bottom'.
    
    Raises:
        AssertionError: If N is not a non-negative integer, position is not 'top' or 'bottom',
                        or N exceeds the number of layers in layer_list.
    
    Example:
        >>> from transformers import BertModel
        >>> model = BertModel.from_pretrained('bert-base-uncased')
        >>> print(len(model.bert.encoder.layer))  # Output: 12
        >>> drop_layers(model.bert.encoder.layer, 3, 'bottom')
        >>> print(len(model.bert.encoder.layer))  # Output: 9
    """
    # Validate inputs
    assert isinstance(N, int) and N >= 0, "N must be a non-negative integer"
    assert position in ['top', 'bottom'], "position must be 'top' or 'bottom'"
    num_layers = len(layer_list)
    assert N <= num_layers, f"Cannot drop {N} layers from a list with {num_layers} layers"
    
    # Drop layers based on position
    if position == 'bottom':
        del layer_list[:N]  # Remove the first N layers
    elif position == 'top':
        del layer_list[-N:]  # Remove the last N layers