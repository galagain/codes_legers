# Poids des configurations des papiers

**105 fichiers de poids distincts, 55,58 Go**. Les alias internes évitent de dupliquer les poids déjà inclus.

Les fichiers regroupent LiPS, TokenMask/EoMT, DynEoMT/VidEoMT, leurs ablations et les initialisations retrouvées localement. Les paramètres entraînés gardent leur type et leur précision. Les états des optimiseurs ont été retirés des checkpoints d’entraînement ; les fichiers préentraînés `.pkl`/`.bin` sont conservés dans leur format original. Chaque extraction a été rechargée et comparée tenseur par tenseur à sa source. Les copies natives ont été vérifiées par SHA-256.

## Couverture

- LiPS : ADE20K et Cityscapes, routages 1–4 niveaux ; ablations R50 à 1–5 couches et 64/128 canaux ; références Mask2Former-R50 ADE20K/Cityscapes.
- TokenMask/EoMT : Tiny, Small, Base et Large, ADE20K panoptique, Cityscapes sémantique, COCO instances et panoptique ; variantes de prédiction/interpolation des masques ; initialisations EoMT Large/Giant.
- DynEoMT/VidEoMT : Small, Base et Large, VIPSeg, OVIS, YTVIS2022 et VSPW ; versions des cibles temporelles employées dans les manuscrits précédents ; sondages linéaire/MLP et ablation de propagation ; initialisations VidEoMT publiques.

## Checkpoints non retrouvés

La collection reste incomplète pour les configurations suivantes :

- ADE20K **sémantique** à 512×512 : EoMT et TokenMask Tiny/Small/Base/Large (8 checkpoints).
- Ablation TokenMask aux strides **1, 8 et 16** : configurations présentes, checkpoints entraînés non retrouvés.
- **TokenMask Giant entraîné** : seul le poids préentraîné EoMT Giant est disponible.
- Références externes **Segmenter** des comparaisons ADE20K/Cityscapes : les checkpoints exacts ne sont pas retrouvés.

Le détail des recherches et les sources des papiers figurent dans [PAPER_COVERAGE.json](PAPER_COVERAGE.json). Aucune autre variante n’est présentée comme un remplacement de ces poids manquants.

## Retrouver la configuration

[manifest.json](manifest.json) donne, pour chaque poids, le chemin source, SHA-256, format, nombre de tenseurs, rôle et configuration associée. `configs/` conserve les configurations originales ; les fichiers JSON TokenMask contiennent les hyperparamètres lisibles. Les exports W&B sont des archives de configuration, pas directement des fichiers CLI. Les trois configurations déjà utilisées par `../run_all.sh` conservent leurs chemins.

Les noms LiPS Cityscapes à 3 et 4 niveaux reprennent ceux des répertoires sources. Attention : leurs configurations archivées indiquent respectivement quatre et trois caractéristiques routées ; cette incohérence des sources est tracée dans `PAPER_COVERAGE.json`.

Vérifier les empreintes depuis `codes_legers/` :

```bash
sha256sum -c weights/SHA256SUMS
```

## LiPS

| Fichier | Rôle | Mo |
|---|---|---:|
| [LiPS/AFFormer_base_ade20k.pth](LiPS/AFFormer_base_ade20k.pth) | AFFormer initialization, required by original LiPS constructor | 36.4 |
| [LiPS/ablations/ade20k_r50_decoder_128channels.pth](LiPS/ablations/ade20k_r50_decoder_128channels.pth) | paper_ablation | 163.4 |
| [LiPS/ablations/ade20k_r50_decoder_1layers.pth](LiPS/ablations/ade20k_r50_decoder_1layers.pth) | paper_ablation | 161.9 |
| [LiPS/ablations/ade20k_r50_decoder_2layers.pth](LiPS/ablations/ade20k_r50_decoder_2layers.pth) | paper_ablation | 164.9 |
| [LiPS/ablations/ade20k_r50_decoder_3layers.pth](LiPS/ablations/ade20k_r50_decoder_3layers.pth) | paper_ablation | 167.8 |
| [LiPS/ablations/ade20k_r50_decoder_4layers.pth](LiPS/ablations/ade20k_r50_decoder_4layers.pth) | paper_ablation | 170.7 |
| [LiPS/ablations/ade20k_r50_decoder_5layers.pth](LiPS/ablations/ade20k_r50_decoder_5layers.pth) | paper_ablation | 173.7 |
| [LiPS/ablations/ade20k_r50_decoder_64channels.pth](LiPS/ablations/ade20k_r50_decoder_64channels.pth) | paper_ablation | 157.4 |
| [LiPS/ade20k_1levels.pth](LiPS/ade20k_1levels.pth) | paper_model | 81.2 |
| [LiPS/ade20k_2levels.pth](LiPS/ade20k_2levels.pth) | paper_model | 80.8 |
| [LiPS/ade20k_3levels.pth](LiPS/ade20k_3levels.pth) | paper_model | 80.0 |
| [LiPS/ade20k_4levels.pth](LiPS/ade20k_4levels.pth) | paper_model | 79.9 |
| [LiPS/ade20k_4levels_best.pth](LiPS/ade20k_4levels_best.pth) | poids déjà inclus | 79.9 |
| [LiPS/baselines/mask2former_r50_ade20k.pkl](LiPS/baselines/mask2former_r50_ade20k.pkl) | paper_baseline | 176.5 |
| [LiPS/baselines/mask2former_r50_cityscapes.pkl](LiPS/baselines/mask2former_r50_cityscapes.pkl) | paper_baseline | 176.3 |
| [LiPS/cityscapes_1levels.pth](LiPS/cityscapes_1levels.pth) | paper_model | 81.1 |
| [LiPS/cityscapes_2levels.pth](LiPS/cityscapes_2levels.pth) | paper_model | 80.6 |
| [LiPS/cityscapes_3levels.pth](LiPS/cityscapes_3levels.pth) | paper_model | 79.7 |
| [LiPS/cityscapes_4levels.pth](LiPS/cityscapes_4levels.pth) | paper_model | 80.2 |

## TokenMask

| Fichier | Rôle | Mo |
|---|---|---:|
| [TokenMask/ade20k_large.pth](TokenMask/ade20k_large.pth) | paper_model | 1233.3 |
| [TokenMask/ade20k_panoptic_base_eomt_feature_cquqkghn.pth](TokenMask/ade20k_panoptic_base_eomt_feature_cquqkghn.pth) | paper_baseline | 374.8 |
| [TokenMask/ade20k_panoptic_base_eomt_logit_971deeg4.pth](TokenMask/ade20k_panoptic_base_eomt_logit_971deeg4.pth) | paper_ablation | 355.8 |
| [TokenMask/ade20k_panoptic_base_eomt_none_5tqbkosy.pth](TokenMask/ade20k_panoptic_base_eomt_none_5tqbkosy.pth) | paper_ablation | 355.8 |
| [TokenMask/ade20k_panoptic_base_tokenmask_logit_y9zhf7tf.pth](TokenMask/ade20k_panoptic_base_tokenmask_logit_y9zhf7tf.pth) | paper_model | 355.8 |
| [TokenMask/ade20k_panoptic_base_tokenmask_none_ctfoekbx.pth](TokenMask/ade20k_panoptic_base_tokenmask_none_ctfoekbx.pth) | paper_ablation | 355.8 |
| [TokenMask/ade20k_panoptic_large_eomt_feature_cvxyrto5.pth](TokenMask/ade20k_panoptic_large_eomt_feature_cvxyrto5.pth) | paper_baseline | 1267.0 |
| [TokenMask/ade20k_panoptic_large_eomt_logit_cqmuavyp.pth](TokenMask/ade20k_panoptic_large_eomt_logit_cqmuavyp.pth) | paper_ablation | 1233.3 |
| [TokenMask/ade20k_panoptic_large_eomt_none_2nu66ri5.pth](TokenMask/ade20k_panoptic_large_eomt_none_2nu66ri5.pth) | paper_ablation | 1233.3 |
| [TokenMask/ade20k_panoptic_large_tokenmask_none_66j1zjzg.pth](TokenMask/ade20k_panoptic_large_tokenmask_none_66j1zjzg.pth) | paper_ablation | 1233.3 |
| [TokenMask/ade20k_panoptic_small_eomt_feature_aoe7zvj0.pth](TokenMask/ade20k_panoptic_small_eomt_feature_aoe7zvj0.pth) | paper_baseline | 96.0 |
| [TokenMask/ade20k_panoptic_small_eomt_logit_ichll3wm.pth](TokenMask/ade20k_panoptic_small_eomt_logit_ichll3wm.pth) | paper_ablation | 91.2 |
| [TokenMask/ade20k_panoptic_small_eomt_none_mkwzj92x.pth](TokenMask/ade20k_panoptic_small_eomt_none_mkwzj92x.pth) | paper_ablation | 91.2 |
| [TokenMask/ade20k_panoptic_small_tokenmask_none_fblc8i1j.pth](TokenMask/ade20k_panoptic_small_tokenmask_none_fblc8i1j.pth) | paper_ablation | 91.2 |
| [TokenMask/ade20k_panoptic_tiny_eomt_feature_oht643kj.pth](TokenMask/ade20k_panoptic_tiny_eomt_feature_oht643kj.pth) | paper_baseline | 25.2 |
| [TokenMask/ade20k_panoptic_tiny_eomt_logit_72mgvuy4.pth](TokenMask/ade20k_panoptic_tiny_eomt_logit_72mgvuy4.pth) | paper_ablation | 24.0 |
| [TokenMask/ade20k_panoptic_tiny_eomt_none_yz9mxnhx.pth](TokenMask/ade20k_panoptic_tiny_eomt_none_yz9mxnhx.pth) | paper_ablation | 24.0 |
| [TokenMask/ade20k_panoptic_tiny_tokenmask_none_5ow5k202.pth](TokenMask/ade20k_panoptic_tiny_tokenmask_none_5ow5k202.pth) | paper_ablation | 24.0 |
| [TokenMask/ade20k_small.pth](TokenMask/ade20k_small.pth) | paper_model | 91.2 |
| [TokenMask/ade20k_tiny.pth](TokenMask/ade20k_tiny.pth) | paper_model | 23.9 |
| [TokenMask/baselines/ade20k_L_640.bin](TokenMask/baselines/ade20k_L_640.bin) | pretrained_baseline | 1267.0 |
| [TokenMask/baselines/ade20k_g_640.bin](TokenMask/baselines/ade20k_g_640.bin) | pretrained_baseline | 4654.8 |
| [TokenMask/cityscapes_semantic_base_eomt_feature_mblq6tc1.pth](TokenMask/cityscapes_semantic_base_eomt_feature_mblq6tc1.pth) | paper_baseline | 381.7 |
| [TokenMask/cityscapes_semantic_base_tokenmask_logit_2dcc4gev.pth](TokenMask/cityscapes_semantic_base_tokenmask_logit_2dcc4gev.pth) | paper_model | 362.8 |
| [TokenMask/cityscapes_semantic_large_eomt_feature_4d3uq2xa.pth](TokenMask/cityscapes_semantic_large_eomt_feature_4d3uq2xa.pth) | paper_baseline | 1276.3 |
| [TokenMask/cityscapes_semantic_large_tokenmask_logit_u13b1a9i.pth](TokenMask/cityscapes_semantic_large_tokenmask_logit_u13b1a9i.pth) | paper_model | 1242.6 |
| [TokenMask/cityscapes_semantic_small_eomt_feature_wecvxrzk.pth](TokenMask/cityscapes_semantic_small_eomt_feature_wecvxrzk.pth) | paper_baseline | 99.5 |
| [TokenMask/cityscapes_semantic_small_tokenmask_logit_bv5thrrj.pth](TokenMask/cityscapes_semantic_small_tokenmask_logit_bv5thrrj.pth) | paper_model | 94.7 |
| [TokenMask/cityscapes_semantic_tiny_eomt_feature_mncmaji2.pth](TokenMask/cityscapes_semantic_tiny_eomt_feature_mncmaji2.pth) | paper_baseline | 26.9 |
| [TokenMask/cityscapes_semantic_tiny_tokenmask_logit_kj915jx2.pth](TokenMask/cityscapes_semantic_tiny_tokenmask_logit_kj915jx2.pth) | paper_model | 25.7 |
| [TokenMask/coco_instance_base_eomt_feature_c41lqjxj.pth](TokenMask/coco_instance_base_eomt_feature_c41lqjxj.pth) | paper_baseline | 374.6 |
| [TokenMask/coco_instance_base_tokenmask_logit_v41oa0w4.pth](TokenMask/coco_instance_base_tokenmask_logit_v41oa0w4.pth) | paper_model | 355.6 |
| [TokenMask/coco_instance_large_eomt_feature_zsr55puj.pth](TokenMask/coco_instance_large_eomt_feature_zsr55puj.pth) | paper_baseline | 1266.7 |
| [TokenMask/coco_instance_large_tokenmask_logit_5cfsfynh.pth](TokenMask/coco_instance_large_tokenmask_logit_5cfsfynh.pth) | paper_model | 1233.0 |
| [TokenMask/coco_instance_small_eomt_feature_gykrved2.pth](TokenMask/coco_instance_small_eomt_feature_gykrved2.pth) | paper_baseline | 95.9 |
| [TokenMask/coco_instance_small_tokenmask_logit_98a41lvj.pth](TokenMask/coco_instance_small_tokenmask_logit_98a41lvj.pth) | paper_model | 91.1 |
| [TokenMask/coco_instance_tiny_eomt_feature_8o0r30wg.pth](TokenMask/coco_instance_tiny_eomt_feature_8o0r30wg.pth) | paper_baseline | 25.1 |
| [TokenMask/coco_instance_tiny_tokenmask_logit_me9e1l90.pth](TokenMask/coco_instance_tiny_tokenmask_logit_me9e1l90.pth) | paper_model | 23.9 |
| [TokenMask/coco_panoptic_base_eomt_feature_la5maj3a.pth](TokenMask/coco_panoptic_base_eomt_feature_la5maj3a.pth) | paper_baseline | 374.7 |
| [TokenMask/coco_panoptic_base_tokenmask_logit_n287tutz.pth](TokenMask/coco_panoptic_base_tokenmask_logit_n287tutz.pth) | paper_model | 355.8 |
| [TokenMask/coco_panoptic_large_eomt_feature_6fy52yaz.pth](TokenMask/coco_panoptic_large_eomt_feature_6fy52yaz.pth) | paper_baseline | 1266.9 |
| [TokenMask/coco_panoptic_large_tokenmask_logit_f3mkbkgr.pth](TokenMask/coco_panoptic_large_tokenmask_logit_f3mkbkgr.pth) | paper_model | 1233.3 |
| [TokenMask/coco_panoptic_small_eomt_feature_orj6e9pq.pth](TokenMask/coco_panoptic_small_eomt_feature_orj6e9pq.pth) | paper_baseline | 96.0 |
| [TokenMask/coco_panoptic_small_tokenmask_logit_kj7dqz6w.pth](TokenMask/coco_panoptic_small_tokenmask_logit_kj7dqz6w.pth) | paper_model | 91.2 |
| [TokenMask/coco_panoptic_tiny_eomt_feature_q3i5oimg.pth](TokenMask/coco_panoptic_tiny_eomt_feature_q3i5oimg.pth) | paper_baseline | 25.1 |
| [TokenMask/coco_panoptic_tiny_tokenmask_logit_1lrzvh5h.pth](TokenMask/coco_panoptic_tiny_tokenmask_logit_1lrzvh5h.pth) | paper_model | 23.9 |

## DynEoMT

| Fichier | Rôle | Mo |
|---|---|---:|
| [DynEoMT/ablations/vipseg_vitl_frozen_linear.pth](DynEoMT/ablations/vipseg_vitl_frozen_linear.pth) | paper_ablation | 1290.8 |
| [DynEoMT/ablations/vipseg_vitl_frozen_mlp.pth](DynEoMT/ablations/vipseg_vitl_frozen_mlp.pth) | paper_ablation | 1291.8 |
| [DynEoMT/ablations/vipseg_vitl_frozen_mlp_no_query_propagation.pth](DynEoMT/ablations/vipseg_vitl_frozen_mlp_no_query_propagation.pth) | paper_ablation | 1291.8 |
| [DynEoMT/baselines/videomt_ovis_base.pth](DynEoMT/baselines/videomt_ovis_base.pth) | paper_baseline | 376.8 |
| [DynEoMT/baselines/videomt_ovis_large.pth](DynEoMT/baselines/videomt_ovis_large.pth) | paper_baseline | 1270.7 |
| [DynEoMT/baselines/videomt_ovis_small.pth](DynEoMT/baselines/videomt_ovis_small.pth) | paper_baseline | 96.4 |
| [DynEoMT/baselines/videomt_vipseg_base.pth](DynEoMT/baselines/videomt_vipseg_base.pth) | paper_baseline | 391.8 |
| [DynEoMT/baselines/videomt_vipseg_large.pth](DynEoMT/baselines/videomt_vipseg_large.pth) | paper_baseline | 1290.8 |
| [DynEoMT/baselines/videomt_vipseg_small.pth](DynEoMT/baselines/videomt_vipseg_small.pth) | paper_baseline | 103.9 |
| [DynEoMT/baselines/videomt_vspw_base.pth](DynEoMT/baselines/videomt_vspw_base.pth) | paper_baseline | 391.8 |
| [DynEoMT/baselines/videomt_vspw_large.pth](DynEoMT/baselines/videomt_vspw_large.pth) | paper_baseline | 1290.8 |
| [DynEoMT/baselines/videomt_vspw_small.pth](DynEoMT/baselines/videomt_vspw_small.pth) | paper_baseline | 103.9 |
| [DynEoMT/baselines/videomt_ytvis2022_base.pth](DynEoMT/baselines/videomt_ytvis2022_base.pth) | paper_baseline | 376.8 |
| [DynEoMT/baselines/videomt_ytvis2022_large.pth](DynEoMT/baselines/videomt_ytvis2022_large.pth) | paper_baseline | 1270.8 |
| [DynEoMT/baselines/videomt_ytvis2022_small.pth](DynEoMT/baselines/videomt_ytvis2022_small.pth) | paper_baseline | 96.4 |
| [DynEoMT/ovis_base.pth](DynEoMT/ovis_base.pth) | paper_model | 377.6 |
| [DynEoMT/ovis_large.pth](DynEoMT/ovis_large.pth) | paper_model | 1271.7 |
| [DynEoMT/ovis_large_best_mcc.pth](DynEoMT/ovis_large_best_mcc.pth) | poids déjà inclus | 1271.7 |
| [DynEoMT/ovis_small.pth](DynEoMT/ovis_small.pth) | paper_model | 96.8 |
| [DynEoMT/pretrained/ovis_vit_large_52.5.pth](DynEoMT/pretrained/ovis_vit_large_52.5.pth) | pretrained_initialization | 1270.7 |
| [DynEoMT/pretrained/videomt/ytvis2019/yt_2019_vit_base_58.2.pth](DynEoMT/pretrained/videomt/ytvis2019/yt_2019_vit_base_58.2.pth) | pretrained_initialization | 376.8 |
| [DynEoMT/pretrained/videomt/ytvis2019/yt_2019_vit_small_52.8.pth](DynEoMT/pretrained/videomt/ytvis2019/yt_2019_vit_small_52.8.pth) | pretrained_initialization | 96.4 |
| [DynEoMT/pretrained/vipseg_vit_large_55.2.pth](DynEoMT/pretrained/vipseg_vit_large_55.2.pth) | pretrained_initialization | 1290.8 |
| [DynEoMT/pretrained/vspw_vit_large_95.0_64.9.pth](DynEoMT/pretrained/vspw_vit_large_95.0_64.9.pth) | pretrained_initialization | 1290.8 |
| [DynEoMT/pretrained/yt_2022_vit_large_42.6.pth](DynEoMT/pretrained/yt_2022_vit_large_42.6.pth) | pretrained_initialization | 1270.8 |
| [DynEoMT/previous_targets/ovis_base.pth](DynEoMT/previous_targets/ovis_base.pth) | earlier_paper_model | 377.6 |
| [DynEoMT/previous_targets/ovis_large.pth](DynEoMT/previous_targets/ovis_large.pth) | earlier_paper_model | 1271.7 |
| [DynEoMT/previous_targets/ovis_small.pth](DynEoMT/previous_targets/ovis_small.pth) | earlier_paper_model | 96.8 |
| [DynEoMT/previous_targets/ytvis2022_base.pth](DynEoMT/previous_targets/ytvis2022_base.pth) | earlier_paper_model | 377.6 |
| [DynEoMT/previous_targets/ytvis2022_large.pth](DynEoMT/previous_targets/ytvis2022_large.pth) | earlier_paper_model | 1271.8 |
| [DynEoMT/previous_targets/ytvis2022_small.pth](DynEoMT/previous_targets/ytvis2022_small.pth) | earlier_paper_model | 96.8 |
| [DynEoMT/vipseg_base.pth](DynEoMT/vipseg_base.pth) | paper_model | 392.6 |
| [DynEoMT/vipseg_large.pth](DynEoMT/vipseg_large.pth) | paper_model | 1291.8 |
| [DynEoMT/vipseg_small.pth](DynEoMT/vipseg_small.pth) | paper_model | 104.3 |
| [DynEoMT/vspw_base.pth](DynEoMT/vspw_base.pth) | paper_model | 392.6 |
| [DynEoMT/vspw_large.pth](DynEoMT/vspw_large.pth) | paper_model | 1291.8 |
| [DynEoMT/vspw_small.pth](DynEoMT/vspw_small.pth) | paper_model | 104.3 |
| [DynEoMT/ytvis2022_base.pth](DynEoMT/ytvis2022_base.pth) | paper_model | 377.6 |
| [DynEoMT/ytvis2022_large.pth](DynEoMT/ytvis2022_large.pth) | paper_model | 1271.8 |
| [DynEoMT/ytvis2022_small.pth](DynEoMT/ytvis2022_small.pth) | paper_model | 96.8 |
