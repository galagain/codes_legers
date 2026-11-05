# Validation du lanceur — 23 octobre 2026

Le lanceur exécute les évaluations complètes des trois modèles sélectionnés,
avec progression en direct et export des métriques renvoyées par les évaluateurs.

- **TokenMask Large : terminé**, 2 000 images ADE20K, PQ = 45,1264 %, SQ = 80,2980 %,
  RQ = 54,1699 %. [Rapport de cette exécution](reports/eval_reelle_TokenMask_20261023/summary.json).
- **LiPS : validation complète en cours**.
  [Log en direct](reports/eval_reelle_LiPS_20261023/LiPS/evaluation.log).
- **DynEoMT : validation complète en cours**, 140 vidéos OVIS / 8 784 images.
  [Log en direct](reports/eval_reelle_DynEoMT_20261023/DynEoMT/evaluation.log).
  Un test technique séparé sur une vidéo de 15 images a terminé avec succès :
  inférence, calcul AP/dynamicité, export JSON et bilan. Ce test limité ne constitue
  pas un score sur OVIS complet ; ses sorties sont dans `/tmp/codes_legers_ovis_integration_result`.

La gestion d’erreur a été vérifiée avec deux interpréteurs volontairement absents :
les deux échecs sont signalés, le lanceur poursuit la seconde méthode et retourne 1,
sans produire de score. L’intégrité des neuf fichiers de poids et la syntaxe Python
ont également été vérifiées.

Les anciens rapports de tests sur quelques images restent conservés dans `reports/`.
Ils décrivent l’ancien lanceur et ne sont pas des évaluations complètes.

## Regroupement des poids — 5 novembre 2026

105 fichiers distincts (55.58 Go) sont indexés dans `weights/manifest.json`. Les ajouts ont été vérifiés par égalité des tenseurs après extraction/rechargement, ou SHA-256 pour les formats natifs. Les configurations et les références des sources sont conservées. Les checkpoints non retrouvés figurent dans `weights/PAPER_COVERAGE.json`. Cette opération n’a pas relancé les évaluations de qualité des modèles.
