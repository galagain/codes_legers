#!/usr/bin/env python3
"""Extract unchanged model tensors; discard optimizer and training state."""
import argparse
import copy
from collections import OrderedDict
from pathlib import Path
import torch


def extract(source, destination, key):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint[key]
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a nonempty tensor state dictionary')
    original_state = state
    state = OrderedDict((k, v.detach().cpu().clone()) for k, v in state.items())
    if hasattr(original_state, '_metadata'):
        state._metadata = copy.deepcopy(original_state._metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: state}, destination)
    restored = torch.load(destination, map_location='cpu', weights_only=True)
    assert restored[key].keys() == checkpoint[key].keys()
    assert getattr(restored[key], '_metadata', None) == getattr(original_state, '_metadata', None)
    for name, tensor in restored[key].items():
        original = checkpoint[key][name]
        assert tensor.dtype == original.dtype and tensor.shape == original.shape
        assert torch.equal(tensor, original), name
    return len(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--key', choices=['model', 'state_dict'], required=True)
    args = parser.parse_args()
    print(f'{extract(args.source, args.destination, args.key)} tensors verified')
