#!/usr/bin/env python3
"""Évalue les trois modèles sélectionnés sur toute la validation, en direct."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT.parent
ENVS = LOCAL / 'codes/panoptic/envs/miniconda3/envs'
MODELS = {
    'LiPS': ('configs/ade20k_4levels.yaml', 'ade20k_4levels_best.pth'),
    'TokenMask': ('configs/ade20k_large.yaml', 'ade20k_large.pth'),
    'DynEoMT': ('configs/DynEoMT/ovis/dinov2/vit-large/dyneomt_online_vitl.yaml', 'ovis_large_best_mcc.pth'),
}


def run(command, cwd, env, log):
    """Affiche immédiatement stdout/stderr et conserve la même sortie dans le log."""
    command = list(map(str, command))
    print('$ ' + shlex.join(command), flush=True)
    with log.open('wb') as stream:
        stream.write(('$ ' + shlex.join(command) + '\n').encode())
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                while chunk := process.stdout.read1(8192):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    stream.write(chunk)
                    stream.flush()
                if process.wait():
                    raise RuntimeError(f'Commande en échec ; voir {log}')
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ade-root', type=Path, default=LOCAL / 'codes/maskdocker/data',
                        help='Racine contenant ADEChallengeData2016 préparé pour Detectron2')
    parser.add_argument('--ade-zips', type=Path, default=LOCAL / 'cyril/codes/lips2/data')
    parser.add_argument('--video-root', type=Path, default=LOCAL / 'DynEOMT/data')
    parser.add_argument('--dynamic-targets', type=Path, default=LOCAL / 'DynEOMT/outputs/dynamic_general_full/labels')
    for method in MODELS:
        parser.add_argument('--python-' + method.lower(), default=os.getenv(method.upper() + '_PYTHON'))
    args = parser.parse_args()
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        parser.error('--device doit avoir la forme cuda:0')
    output = (args.output or ROOT / 'reports' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for method in args.models:
        config, weight = MODELS[method]
        folder = output / method
        folder.mkdir()
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
                   DYNEOMT_SKIP_PRETRAINED='1', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES=args.device[5:], PYTHONPATH=str(ROOT / method))
        python = getattr(args, 'python_' + method.lower()) or str(
            ENVS / ('eomt' if method == 'TokenMask' else 'eomt2') / 'bin/python')
        if Path(python) == ENVS / 'eomt2/bin/python':
            env['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6' + (
                ' ' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        metrics = folder / 'metrics.json'
        command = [python, '-u', ROOT / 'tools/evaluate_model.py', method, metrics]
        print(f'\n=== {method} : {weight} — validation complète ===', flush=True)
        summary[method] = {'status': 'RUNNING', 'checkpoint': weight}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        try:
            if method == 'LiPS':
                extension = folder / 'extension.json'
                run([python, ROOT / 'tools/prepare_msda.py', '--result', extension], ROOT, env, folder / 'extension.log')
                env['PYTHONPATH'] += os.pathsep + json.loads(extension.read_text())['path']
            if method == 'TokenMask':
                command += ['validate', '-c', config, '--data.path', args.ade_zips.resolve(),
                            '--data.batch_size', '1', '--data.num_workers', '2',
                            '--trainer.devices', '[0]', '--trainer.logger', 'false',
                            '--model.ckpt_path', ROOT / 'weights' / method / weight]
            else:
                env['DETECTRON2_DATASETS'] = str((args.ade_root if method == 'LiPS' else args.video_root).resolve())
                command += ['--eval-only', '--config-file', config, 'MODEL.WEIGHTS',
                            ROOT / 'weights' / method / weight, 'OUTPUT_DIR', folder,
                            'DATALOADER.NUM_WORKERS', '2']
                if method == 'DynEoMT':
                    command += ['MODEL.DYNAMIC.TARGET_ROOT', args.dynamic_targets.resolve()]
            run(command, ROOT / method, env, folder / 'evaluation.log')
            measured = json.loads(metrics.read_text())
            if not measured:
                raise RuntimeError('Évaluation terminée sans métriques.')
            summary[method] = {'status': 'OK', 'checkpoint': weight, 'metrics': measured}
        except KeyboardInterrupt:
            summary[method] = {'status': 'INTERRUPTED', 'checkpoint': weight}
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print('\nÉvaluation interrompue.', flush=True)
            return 130
        except (OSError, RuntimeError, ValueError) as error:
            print(f'[ÉCHEC] {method} : {error}', flush=True)
            summary[method] = {'status': 'FAIL', 'error': str(error)}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print('\n=== Bilan des évaluations de cette exécution ===', flush=True)
    for method, result in summary.items():
        print(f'{method} : {result["status"]}')
        for key, value in result.get('metrics', {}).get('scores', {}).items():
            print(f'  {key}: {value:.4f}' if value is not None else f'  {key}: non défini')
    print(f'Rapport : {output / "summary.json"}', flush=True)
    return int(any(result['status'] != 'OK' for result in summary.values()))


if __name__ == '__main__':
    raise SystemExit(main())
