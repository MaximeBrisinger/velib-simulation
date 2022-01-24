# velib-simulation

Ce code simule un systeme de velib à 5 stations. Deux cas d'études sont envisagés :
    - Un seul vélo est présent dans le système.
    - 91 vélos sont présents dans le systèmes

## Execution du code

Deux modes d'exécution sont disponibles :

- Pour l'étude du cas à un seul vélo : exécuter dans un terminal après s'être placé dans le dossier src

        python3 main.py -i 1_velo

- Pour l'étude du cas réel :

        python3 main.py -i donnees

## Fichiers de codes

Le code présente deux fichiers `.py` :

- `fill_data.py` permet de générer les fichiers de données au format `.npy`.
- `main.py` est le fichier contenant les fonctions permettant la simulation.

Détail des fonctions présentes dans `main.py` :

- `temps_d_attente` tire le temps d'attente avant le prochain changeemnt d'état.
- `nouvel_etat` tire le prochain état du système.
- `main` réalise la simulation.
- `visualisation` permet d'afficher sommairement la solution grace à la bibliothèque networkx.
- `proba_stationnaire_1_velo` calcule la probabilité stationnaire dans le cas où un seul vélo est présent dans le système.