# LiPS, TokenMask et DynEoMT — codes allégés

> **Version GitHub : les poids ne sont pas versionnés.** Le dossier `weights/`
> contient leur index, leur provenance et les configurations. Pour exécuter les
> modèles, replacer les checkpoints aux chemins indiqués dans
> [weights/manifest.json](weights/manifest.json). Les rapports d’évaluation et les
> caches restent également locaux.

Les codes sont séparés dans `LiPS/`, `TokenMask/` et `DynEoMT/`.
Les poids sont regroupés dans `weights/` (environ 55,58 Go, précision conservée,
états d’optimiseur supprimés). Les datasets et environnements restent à l’extérieur.


La collection des poids a été étendue aux configurations des papiers, à leurs
ablations et aux modèles de référence retrouvés. Voir [weights/README.md](weights/README.md)
pour l’index complet et la liste explicite des checkpoints non retrouvés.

## Évaluer les trois modèles

Sur cette machine, les environnements, données et annotations sont déjà repérés :

```bash
cd /data1/is156025/cg276974/codes_legers
./run_all.sh
```

Le script charge les poids et lance successivement les **évaluations complètes** :

| Méthode | Poids évalués | Validation | Métriques calculées |
|---|---|---|---|
| LiPS | `weights/LiPS/ade20k_4levels_best.pth` | ADE20K, 2 000 images | PQ, SQ, RQ, mIoU, AP |
| TokenMask Large | `weights/TokenMask/ade20k_large.pth` | ADE20K, 2 000 images | PQ, SQ, RQ, objets et fond |
| DynEoMT Large | `weights/DynEoMT/ovis_large_best_mcc.pth` | OVIS, 140 vidéos (8 784 images) | AP, AP50, AP75, Dyn-MCC, Dyn-F1, Dyn-AUROC… |

La progression des évaluateurs est affichée **en direct dans le terminal**.
Les scores apparaissent à la fin de chaque évaluation, puis dans un bilan commun.
**Chaque score affiché est recalculé pendant cette exécution.** Le lanceur ne lit
aucun résultat historique ni tableau de papier. Aucun sous-ensemble n’est utilisé.

Chaque lancement crée `reports/<date_heure>/` : un `evaluation.log` et un
`metrics.json` par méthode, puis `summary.json` pour le bilan. Les évaluateurs
conservent aussi leurs sorties natives. Un échec est signalé et donne un code de
sortie 1 ; les autres méthodes sont quand même évaluées. Une validation complète
prend plusieurs minutes pour ADE20K et peut prendre plusieurs heures pour OVIS.

```bash
# Un seul modèle :
./run_all.sh --models TokenMask

# Choisir un GPU et deux modèles :
./run_all.sh --device cuda:2 --models LiPS DynEoMT

# Toutes les options :
./run_all.sh --help
```

## Chemins des données et environnements

Les valeurs par défaut correspondent aux fichiers présents sur cette machine.
Pour utiliser le dossier ailleurs, installer les dépendances en suivant
[INSTALLATION.md](INSTALLATION.md), puis préciser les chemins :

```bash
./run_all.sh \
  --python-lips /chemin/env_lips/bin/python \
  --python-tokenmask /chemin/env_tokenmask/bin/python \
  --python-dyneomt /chemin/env_dyneomt/bin/python \
  --ade-root /chemin/datasets_detectron2 \
  --ade-zips /chemin/archives_ade20k \
  --video-root /chemin/datasets_video \
  --dynamic-targets /chemin/labels
```

- `--ade-root` contient `ADEChallengeData2016/`, ses images, ses annotations
  sémantiques Detectron2, panoptiques et instances préparées.
- `--ade-zips` contient `ADEChallengeData2016.zip` et `annotations_instance.zip`.
- `--video-root` contient `ovis/valid/` et `ovis/annotations/annotations_valid.json`.
- `--dynamic-targets` contient `ovis/` et les cibles temporelles corrigées utilisées
  pour mesurer la dynamicité. Sur cette machine : `DynEOMT/outputs/dynamic_general_full/labels`.

L’extension CUDA LiPS est réutilisée ou compilée automatiquement dans
`~/.cache/codes_legers/`. Le toolkit peut être indiqué avec `CUDA_HOME`.

Les variantes plus petites restent disponibles dans `weights/` ; les configurations
et la provenance des checkpoints sont dans [weights/manifest.json](weights/manifest.json).
Les instructions d’inférence et d’entraînement manuelles sont dans
[INSTALLATION.md](INSTALLATION.md). Les sources d’origine et modifications locales
sont tracées dans [provenance.json](provenance.json).
