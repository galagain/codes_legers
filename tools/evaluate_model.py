#!/usr/bin/env python3
"""Utilise les évaluateurs des projets et sauvegarde uniquement leur résultat courant."""
import importlib
import json
import logging
import math
from pathlib import Path
import sys
import warnings

# Cet avertissement est émis à chaque image par le backbone LiPS d'origine.
warnings.filterwarnings('ignore', message=r'`nn.functional.upsample` is deprecated.*')


class EvaluationLogs(logging.Filter):
    def filter(self, record):
        # La configuration complète est déjà sauvegardée par Detectron2 dans config.yaml.
        return not record.getMessage().startswith((
            'Environment info:', 'Command line arguments:', 'Contents of args.config_file=',
            'Running with full config:', 'Model:\n',
        ))


def save(result, method, destination):
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        value = float(value)
        return value if math.isfinite(value) else None

    result = clean(result)
    scores = {}
    if method == 'TokenMask':
        for key, value in result[0].items():
            if key.startswith('metrics/val_'):
                scores[key.removeprefix('metrics/val_').upper() + ' (%)'] = value * 100 if value is not None else None
    else:
        for task, values in result.items():
            for key, value in values.items():
                if key in {'PQ', 'PQ_th', 'PQ_st', 'SQ', 'RQ', 'mIoU', 'mACC', 'pACC',
                           'AP', 'AP50', 'AP75', 'AR1', 'AR10'} or task == 'dynamic':
                    scores[f'{task}/{key}'] = value
    if not scores or not any(value is not None for value in scores.values()):
        raise RuntimeError('Aucune métrique mesurée exploitable.')
    payload = {'scope': 'full_validation', 'scores': scores, 'raw': result}
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'\n=== {method} : métriques mesurées sur toute la validation ===', flush=True)
    for key, value in scores.items():
        print(f'{key}: {value:.4f}' if value is not None else f'{key}: non défini', flush=True)


if __name__ == '__main__':
    method, destination = sys.argv[1], Path(sys.argv[2])
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    if method == 'TokenMask':
        entrypoint = importlib.import_module('main')

        class EvaluationCLI(entrypoint.LightningCLI):
            def validate(self, model, **kwargs):
                result = self.trainer.validate(model, **kwargs)
                save(result, method, destination)

        entrypoint.LightningCLI = EvaluationCLI
        entrypoint.cli_main()
    else:
        entrypoint = importlib.import_module('train_net' if method == 'LiPS' else 'train_net_video')
        for name in ('detectron2', 'detectron2.engine.defaults'):
            logging.getLogger(name).addFilter(EvaluationLogs())
        from detectron2.engine import default_argument_parser
        result = entrypoint.main(default_argument_parser().parse_args())
        save(result, method, destination)
