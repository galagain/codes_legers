# Installation et commandes manuelles

Pour évaluer les trois modèles avec une seule commande, voir [README.md](README.md).

## Préparation commune

Linux, GPU NVIDIA et Conda sont recommandés. LiPS nécessite aussi un compilateur C++
et le toolkit CUDA (`nvcc`) pour son attention déformable. Installer les dépendances
dans des environnements séparés. Les bibliothèques et poids ne sont pas inclus dans
les dossiers de code ; la première installation nécessite Internet.

Depuis ce dossier :

```bash
export BUNDLE="$PWD"
sha256sum -c weights/SHA256SUMS
export WANDB_MODE=disabled
```

Les commandes suivantes supposent que `BUNDLE` reste défini. Pour choisir une carte,
préfixer une commande avec `CUDA_VISIBLE_DEVICES=0` (ou un autre numéro disponible).

## 1. LiPS

### Installation

Le code d'origine utilisait Python 3.8 / PyTorch 1.9. La copie inclut une adaptation
des accès aux tenseurs C++ pour compiler aussi avec PyTorch 2.7, sans changer le calcul.

```bash
conda create -n lips_light python=3.11 -y
conda activate lips_light
cd "$BUNDLE/LiPS"
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833'
python -m pip install 'git+https://github.com/cocodataset/panopticapi.git'

# Adapter au toolkit installé ; CUDA 12.6 correspond à la roue PyTorch ci-dessus.
export CUDA_HOME=/usr/local/cuda
cd mask2former/modeling/pixel_decoder/ops
python -m pip install --no-build-isolation .
cd "$BUNDLE/LiPS"
```

L'extension est compilée pour le GPU visible. Sur une machine de compilation sans
GPU, ajouter `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0` pour une cible A100, ou remplacer
`8.0` par l'architecture du GPU cible. Ne pas recopier les anciens binaires `.so`.

### Inférence sur une image

```bash
cd "$BUNDLE/LiPS"
mkdir -p outputs/demo
PYTHONPATH=. python demo/demo.py \
  --config-file configs/ade20k_4levels.yaml \
  --input /chemin/image.jpg --output outputs/demo \
  --opts MODEL.WEIGHTS ../weights/LiPS/ade20k_4levels.pth
```

Le lien relatif `LiPS/checkpoints/AFFormer_base_ade20k.pth` pointe vers le dossier
commun de poids. Il est nécessaire même à l'inférence, car le code original charge
cette initialisation à la construction, puis les poids LiPS complets.

### Données, évaluation et entraînement

LiPS lit ADE20K décompressé et ses annotations panoptiques au format Detectron2.
Voir [LiPS/datasets/README.md](LiPS/datasets/README.md) pour la préparation.
Avec `ADEChallengeData2016/images`, `annotations` et `annotations_instance` déjà
présents sous la racine, les scripts conservés créent les annotations nécessaires :

```bash
export DETECTRON2_DATASETS=/chemin/datasets
cd "$BUNDLE/LiPS"
python datasets/prepare_ade20k_sem_seg.py
python datasets/prepare_ade20k_ins_seg.py
python datasets/prepare_ade20k_pan_seg.py

python train_net.py --num-gpus 1 --eval-only \
  --config-file configs/ade20k_4levels.yaml \
  MODEL.WEIGHTS ../weights/LiPS/ade20k_4levels.pth OUTPUT_DIR outputs/eval

# Nouveau fine-tuning ; ne pas ajouter --resume avec les poids allégés.
python train_net.py --num-gpus 1 \
  --config-file configs/ade20k_4levels.yaml \
  MODEL.WEIGHTS ../weights/LiPS/ade20k_4levels.pth \
  SOLVER.IMS_PER_BATCH 2 OUTPUT_DIR outputs/finetune
```

La recette historique utilise un batch total de 16 et 160 000 itérations.
Pour le meilleur checkpoint ADE20K retenu, remplacer le fichier de poids par
`../weights/LiPS/ade20k_4levels_best.pth` ; la configuration reste identique.
Le batch de 2 est un exemple de relance plus économe, pas une reproduction exacte
du protocole. Pour repartir de l'initialisation AFFormer, remplacer le chemin
`MODEL.WEIGHTS` par une chaîne vide `''`.

## 2. TokenMask

### Installation

```bash
conda create -n tokenmask_light python=3.13.2 -y
conda activate tokenmask_light
cd "$BUNDLE/TokenMask"
python -m pip install -r requirements.txt
```

Les versions proviennent du dépôt EoMT local (PyTorch 2.7, Lightning 2.5.1.post0,
timm 1.0.15). Les recettes utilisent un logger CSV local ; aucun compte W&B requis.

### Évaluation et fine-tuning

Le dossier de données doit contenir **sans décompression** `ADEChallengeData2016.zip`
et `annotations_instance.zip`. Le second ZIP doit avoir un dossier racine
`annotations_instance/` contenant `training/` et `validation/`.
Sur cette machine, ces archives se trouvent dans
`/data1/is156025/cg276974/cyril/codes/lips2/data`.

```bash
export ADE_ZIPS=/chemin/archives_ade20k
cd "$BUNDLE/TokenMask"

# Un batch pour vérifier la relance ; retirer limit_val_batches pour tout val.
python main.py validate -c configs/ade20k_tiny.yaml \
  --data.path "$ADE_ZIPS" --data.batch_size 1 --data.num_workers 0 \
  --trainer.devices 1 --trainer.limit_val_batches 1 \
  --model.ckpt_path ../weights/TokenMask/ade20k_tiny.pth \
  --model.network.masked_attn_enabled False

# Exemple Small : même commande avec ade20k_small.yaml et ade20k_small.pth.
python main.py fit -c configs/ade20k_small.yaml \
  --data.path "$ADE_ZIPS" --data.batch_size 2 --trainer.devices 1 \
  --model.ckpt_path ../weights/TokenMask/ade20k_small.pth \
  --trainer.max_epochs 1 --compile_disabled
```

Pour Large, utiliser `configs/ade20k_large.yaml` et
`../weights/TokenMask/ade20k_large.pth`. Un batch de 1 convient aux tests rapides.

Utiliser **`--model.ckpt_path`**, et non `--ckpt_path` : les fichiers ne contiennent
plus l'état de reprise Lightning. Pour entraîner depuis le backbone préentraîné,
omettre `--model.ckpt_path` ; timm téléchargera alors son initialisation si elle
n'est pas déjà en cache. L'indexation initiale des ZIP peut prendre quelques minutes,
y compris lorsqu'on demande un seul batch.

## 3. DynEoMT

### Installation

```bash
conda create -n dyneomt_light python=3.12.3 -y
conda activate dyneomt_light
cd "$BUNDLE/DynEoMT"
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833'
python -m pip install 'git+https://github.com/cocodataset/panopticapi.git'
python -m pip install -r requirements.txt
export WANDB_MODE=disabled
```

Les dépendances d'export ONNX/TensorRT ont été retirées. Le code PyTorch et les
recettes VidEoMT/DynEoMT S/B/L pour VIPSeg, OVIS, YouTube-VIS 2022 et VSPW sont conservés.
Deux modèles S et le modèle OVIS-L sélectionné sont fournis ; les autres recettes
requièrent leurs propres poids.

### Inférence vidéo, sans annotations ni pseudo-labels

Placer les images d'une séquence dans un dossier, avec des noms triables dans
l'ordre temporel. Le script suivant fonctionne sur un GPU NVIDIA classique aussi :

```bash
cd "$BUNDLE/DynEoMT"
DYNEOMT_SKIP_PRETRAINED=1 PYTHONPATH=. python scripts/run_jetson_demo.py \
  --config configs/DynEoMT/VIPSeg/dinov2/vit-small/dyneomt_online_vits.yaml \
  --checkpoint ../weights/DynEoMT/vipseg_small.pth \
  --input-dir /chemin/sequence --output-dir outputs/demo \
  --max-frames 30 --device cuda:0
```

`DYNEOMT_SKIP_PRETRAINED=1` évite un téléchargement DINOv2 inutile lorsque les poids
complets sont fournis. La démo produit une vidéo annotée et des résultats JSON.
Cette démo de rendu est spécifique à la sortie panoptique VIPSeg ; utiliser les
évaluateurs ou `scripts/visualize_dyneomt.py` pour OVIS.

### Évaluation et fine-tuning

Préparer les données selon [DynEoMT/datasets/README.md](DynEoMT/datasets/README.md).
La racine locale existante est `/data1/is156025/cg276974/DynEOMT/data`.

```bash
cd "$BUNDLE/DynEoMT"
export DETECTRON2_DATASETS=/chemin/datasets_video

# Segmentation seule : les pseudo-labels de dynamicité ne sont pas requis.
DYNEOMT_SKIP_PRETRAINED=1 python train_net_video.py --num-gpus 1 --eval-only \
  --config-file configs/DynEoMT/VIPSeg/dinov2/vit-small/dyneomt_online_vits.yaml \
  MODEL.WEIGHTS ../weights/DynEoMT/vipseg_small.pth \
  MODEL.DYNAMIC.REQUIRED False MODEL.DYNAMIC.EVAL_ENABLED False \
  OUTPUT_DIR outputs/eval_vipseg

# Fine-tuning avec les pseudo-labels déjà construits.
DYNEOMT_SKIP_PRETRAINED=1 python train_net_video.py --num-gpus 1 \
  --config-file configs/DynEoMT/VIPSeg/dinov2/vit-small/dyneomt_online_vits.yaml \
  MODEL.WEIGHTS ../weights/DynEoMT/vipseg_small.pth \
  MODEL.DYNAMIC.TARGET_ROOT /chemin/dynamic_targets/labels \
  OUTPUT_DIR outputs/finetune_vipseg
```

Pour OVIS, remplacer `VIPSeg` par `ovis` dans la configuration et utiliser
`../weights/DynEoMT/ovis_small.pth`. Pour évaluer aussi la dynamicité, garder
`MODEL.DYNAMIC.EVAL_ENABLED True`, `MODEL.DYNAMIC.REQUIRED True` et fournir
`MODEL.DYNAMIC.TARGET_ROOT`.

Pour OVIS-L, choisir
`configs/DynEoMT/ovis/dinov2/vit-large/dyneomt_online_vitl.yaml` et
`../weights/DynEoMT/ovis_large_best_mcc.pth`.

La construction des pseudo-labels est indépendante de l'inférence. Le constructeur
est conservé ; MemFlow et ses poids restent externes pour ne pas alourdir le dossier :

```bash
git clone https://github.com/DQiaole/MemFlow.git external/MemFlow
# Installer ses dépendances et placer MemFlowNet_sintel.pth dans external/MemFlow/ckpts/.
python scripts/build_dynamic_targets.py \
  --data-root "$DETECTRON2_DATASETS" --datasets vipseg --devices 0 --dry-run
python scripts/build_dynamic_targets.py \
  --data-root "$DETECTRON2_DATASETS" --datasets vipseg --devices 0 \
  --output-root outputs/dynamic_targets \
  --memflow-repo external/MemFlow \
  --checkpoint external/MemFlow/ckpts/MemFlowNet_sintel.pth
```

Le checkpoint MemFlow se récupère dans la release v1.0.0 du dépôt MemFlow.
Les labels sont écrits sous `outputs/dynamic_targets/labels/<dataset>/<sequence>/targets.json`.

