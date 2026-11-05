#!/usr/bin/env python3
"""Reuse or build LiPS CUDA extension in a cache, never in the source bundle."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
torch.cuda.set_device(args.device)
try:
    extension = importlib.import_module('MultiScaleDeformableAttention')
except ImportError:
    from torch.utils.cpp_extension import CUDA_HOME
    if not CUDA_HOME:
        raise RuntimeError('CUDA toolkit/nvcc absent : définir CUDA_HOME pour compiler LiPS.')
    CUDA_HOME = str(Path(CUDA_HOME).resolve())
    source = Path(__file__).resolve().parents[1] / 'LiPS/mask2former/modeling/pixel_decoder/ops'
    digest = hashlib.sha256()
    architecture = os.environ.get('TORCH_CUDA_ARCH_LIST') or '.'.join(map(str, torch.cuda.get_device_capability()))
    digest.update(f'{sys.executable}|{torch.__version__}|{CUDA_HOME}|{architecture}'.encode())
    for file in sorted(source.rglob('*')):
        if file.is_file():
            digest.update(str(file.relative_to(source)).encode())
            digest.update(file.read_bytes())
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'codes_legers/msda' / digest.hexdigest()[:20]
    lib = cache / 'lib'
    lib.mkdir(parents=True, exist_ok=True)
    if not list(lib.glob('MultiScaleDeformableAttention*.so')):
        shutil.copytree(source, cache / 'source', dirs_exist_ok=True)
        env = os.environ.copy()
        env.setdefault('MAX_JOBS', '2')
        env['TORCH_CUDA_ARCH_LIST'] = architecture
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-temp', str(cache / 'build'), '--build-lib', str(lib)], cwd=cache / 'source', env=env, check=True)
    sys.path.insert(0, str(lib))
    extension = importlib.import_module('MultiScaleDeformableAttention')
directory = str(Path(extension.__file__).resolve().parent)
args.result.write_text(json.dumps({'path': directory, 'python': sys.executable, 'torch': torch.__version__}, indent=2))
print(f'Extension CUDA importée : {extension.__file__}')
