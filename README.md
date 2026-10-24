# brute-force-btc
un outil de brute force 


# Puzzle BTC GPU

Recherche de clés privées Bitcoin (secp256k1) **dans une plage connue**, sur **toutes les cartes NVIDIA disponibles** (multi-GPU), avec repli automatique sur CPU et **reprise après coupure**. Pensé pour Google Colab et Kaggle, utilisable sur n'importe quelle machine Linux.

Le projet tient en **un seul script : `puzzle.py`**. Il contient tout le code (C++ / CUDA), le compile au premier lancement puis lance la recherche. Rien d'autre à installer que Python 3, `g++` et, pour le GPU, CUDA (`nvcc`), déjà présents sur Colab et Kaggle.

> **À lire d'abord.** Cet outil ne « casse » rien : il teste toutes les clés d'une plage réduite, comme dans les puzzles Bitcoin publics (clé cachée entre 2^65 et 2^66, par exemple). L'espace complet de 256 bits est hors de portée de n'importe quelle machine. À utiliser pour les puzzles publics et pour retrouver **vos propres** clés. Aucune garantie.

## Sommaire
1. [Démarrage rapide (Colab / Kaggle)](#1-démarrage-rapide-colab--kaggle)
2. [Sur votre machine](#2-sur-votre-machine)
3. [Utilisation](#3-utilisation)
4. [Options](#4-options)
5. [Exemples](#5-exemples)
6. [Plusieurs cibles](#6-plusieurs-cibles-fichier-dadresses)
7. [Reprise après coupure](#7-reprise-après-coupure)
8. [Résultats et codes de retour](#8-résultats-et-codes-de-retour)
9. [Dépannage](#9-dépannage)
10. [Fonctionnement et limites](#10-fonctionnement-et-limites)

---

## 1. Démarrage rapide (Colab / Kaggle)

1. **Activez un GPU** :
   - Colab : *Exécution → Modifier le type d'exécution → GPU*
   - Kaggle : *Settings → Accelerator → GPU* (T4 ×2 : les deux cartes sont utilisées) et *Internet : On* (nécessaire pour télécharger le script)
2. Dans une cellule :

```python
!wget -q https://raw.githubusercontent.com/VOTRE_COMPTE/VOTRE_DEPOT/main/puzzle.py
!python puzzle.py --build        # compilation, quelques minutes, une seule fois
!python puzzle.py --gpuinfo      # vérifie les cartes détectées
```

3. Lancez la recherche (exemple de test, clé de 32 bits) :

```python
!python puzzle.py $(python puzzle.py --genpub C3A5B7D1) 32
```

Pour votre vraie recherche, remplacez la cible et la plage :

```python
!python puzzle.py 03a34b99f22c790c4e36b2b3c2c35a36db06226e41c692fc82b8b56ac1c540c5bd 66
```

> Au premier lancement, le programme exécute des **auto-tests sur chaque carte**. Si une carte échoue, elle est ignorée et le calcul continue (CPU ou autres cartes) : le résultat reste fiable.

**Sessions qui s'arrêtent.** Relancez simplement la même commande : la progression est reprise (voir [section 7](#7-reprise-après-coupure)). Sur Colab, montez Google Drive et placez-y les résultats et la reprise :

```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/puzzle
!python puzzle.py <cible> 70 -out /content/drive/MyDrive/puzzle/cle.txt -ckpt /content/drive/MyDrive/puzzle/reprise.ckpt
```

## 2. Sur votre machine

Prérequis : Linux, Python 3, `g++` (C++17). Pour le GPU : pilote NVIDIA + CUDA Toolkit (`nvcc`).

```bash
wget https://raw.githubusercontent.com/VOTRE_COMPTE/VOTRE_DEPOT/main/puzzle.py
python3 puzzle.py --build          # GPU si nvcc est présent, sinon CPU seul
python3 puzzle.py --gpuinfo        # cartes détectées
python3 puzzle.py <cible> <bits>   # recherche
```

Le binaire est compilé une fois et mis en cache dans `~/.cache/puzzle-gpu/` (changez avec la variable d'environnement `PUZZLE_CACHE`) ; les lancements suivants sont immédiats. Les fichiers produits (résultats, reprise, filtres `.bloom`) sont écrits dans le dossier courant.

| Commande du script | Effet |
|---|---|
| `--build` | compile si nécessaire (GPU si possible) |
| `--rebuild` | force la recompilation (utile après une mise à jour de CUDA ou un changement de carte) |
| `--build-cpu` | compile la version CPU seule |
| `--extract DOSSIER` | écrit les fichiers source (`puzzle.cpp`, `gpu.cu`, …) pour les lire ou les modifier |
| `--help` | aide |

Si `nvcc` est absent ou échoue, le script se replie automatiquement sur le CPU ; si `nvcc` apparaît plus tard, il recompile en version GPU. Le journal de compilation est dans le dossier de cache (`build.log`). Windows : utilisez WSL2.

## 3. Utilisation

```
python3 puzzle.py <cible> <bits | début:fin> [address] [mode] [options]
```

| Argument | Description |
|---|---|
| `<cible>` | Clé publique compressée (`02…` / `03…`, 66 caractères hex), adresse Bitcoin, fichier `.txt` (une cible par ligne) ou fichier `.bloom` |
| `<bits>` | Nombre de bits : la clé est dans `[2^(bits-1), 2^bits − 1]` (ex. `66`) |
| `<début:fin>` | Plage hexadécimale quelconque (ex. `2000000:3ffffff`) |
| `address` | À ajouter quand la cible est une adresse (`1…` ou `bc1q…`) |

**Mode de parcours** (un seul par lancement ; tous les calculateurs travaillent sur la même suite de blocs, sans bloc refait) :

| Option | Mode |
|---|---|
| `-r` | aléatoire sans répétition (défaut) |
| `-s` | séquentiel |
| `-sobol` | suite de Sobol |
| `-pi`, `-e`, `-ln` | suites basées sur π, e, ln 2 |
| `-m` | Monte Carlo par zones |

Tous les modes couvrent la plage entière exactement une fois ; seul l'ordre change (aucun n'a de meilleure chance statistique).

Utilitaires :

```bash
python3 puzzle.py --genpub <clé_privée_hex>     # clé publique compressée
python3 puzzle.py --genaddr <clé_privée_hex>    # adresses 1... et bc1q...
python3 puzzle.py --gpuinfo                     # cartes CUDA détectées
```

Cibles gérées : **P2PKH** (`1…`) et **P2WPKH** (`bc1q…`) de clés **compressées**, ou clé publique compressée directe. P2SH (`3…`), Taproot (`bc1p…`) et clés non compressées ne sont pas gérés.

## 4. Options

| Option | Effet | Défaut |
|---|---|---|
| `-nogpu` | CPU uniquement | — |
| `-gpus 0,1` | Utiliser uniquement ces cartes | toutes |
| `-t N` | N cœurs CPU **en plus** des GPU | 0 avec GPU, tous sans GPU |
| `-ckpt FICHIER` | Fichier de reprise | `puzzle_<empreinte>.ckpt` |
| `-every S` | Période de sauvegarde de la reprise (secondes) | 60 |
| `-fresh` | Ignorer une reprise existante, repartir de zéro | — |
| `-nockpt` | Ne pas sauvegarder la progression | — |
| `-out FICHIER` | Fichier des résultats | `cle_trouvee.txt` |
| `-log S` | Période du journal hors terminal (notebooks, redirection) | 10 |
| `-gjb N` | Lots par travail GPU (réglage avancé) | 16 |

Sur Colab (2 cœurs CPU), laissez `-t` à sa valeur par défaut : le CPU apporte très peu face au GPU.

## 5. Exemples

```bash
# Clé publique, plage de 66 bits, toutes les cartes
python3 puzzle.py 03a34b99f22c790c4e36b2b3c2c35a36db06226e41c692fc82b8b56ac1c540c5bd 66

# Adresse, plage hexadécimale, mode séquentiel
python3 puzzle.py 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH 400000000:7ffffffff address -s

# Seulement la 2e carte
python3 puzzle.py <cible> 70 -gpus 1

# CPU seul, 4 cœurs, mode Sobol
python3 puzzle.py <cible> 40 -nogpu -t 4 -sobol

# Test rapide de l'installation (clé connue)
python3 puzzle.py $(python3 puzzle.py --genpub C3A5B7D1) 32
```

## 6. Plusieurs cibles (fichier d'adresses)

Un fichier texte, une cible par ligne (le premier mot de chaque ligne compte, le reste est ignoré ; doublons et lignes invalides écartés avec un résumé) :

```bash
python3 puzzle.py cibles.txt 66 address       # fichier d'adresses
python3 puzzle.py cibles.txt 66               # fichier de clés publiques
```

Au premier lancement, le programme construit et enregistre `cibles.bloom` (filtre de Bloom + liste exacte, avec somme de contrôle). Les lancements suivants le réutilisent tant que `cibles.txt` ne change pas ; vous pouvez aussi passer directement `cibles.bloom`. Le filtre est copié en mémoire de chaque carte (de 4 à 8 octets par cible environ) : vérifiez qu'il tient.

## 7. Reprise après coupure

La progression est sauvegardée toutes les 60 s (écriture atomique) et à l'arrêt par Ctrl+C / bouton Stop. **Relancez exactement la même commande** : la reprise est automatique et ne refait (presque) rien, même si le nombre de cartes a changé.

- Le fichier est lié à la recherche (cible, plage, mode) : un fichier d'une autre recherche ou corrompu est refusé avec un message.
- Colab et Kaggle effacent le disque à la fin de la session : placez `-ckpt` et `-out` sur Google Drive (Colab) ou téléchargez les fichiers avant l'arrêt.
- La reprise est supprimée à la fin d'une recherche terminée.

## 8. Résultats et codes de retour

Chaque clé trouvée est écrite dans **deux fichiers** (`cle_trouvee.txt` et `cle_trouvee_copie.txt`, ou `-out` et `-out.copie`), relue pour vérification, avec emplacements de secours si le disque échoue. Elle est aussi affichée à l'écran.

| Code | Signification |
|---|---|
| 0 | toutes les cibles trouvées |
| 1 | plage entièrement parcourue, au moins une cible non trouvée |
| 2 | échec des auto-tests |
| 4 | clé trouvée mais impossible à enregistrer (recopiez-la depuis l'écran) |
| 130 | interrompu (progression sauvegardée) |

Toute clé signalée par le GPU est **revérifiée sur CPU** par un chemin de calcul indépendant avant d'être enregistrée.

## 9. Dépannage

| Symptôme | Cause / solution |
|---|---|
| `nvcc introuvable : version CPU seule` | GPU non activé dans le notebook, ou CUDA Toolkit absent. Activez l'accélérateur puis `python puzzle.py --rebuild` |
| `Aucune carte NVIDIA utilisable` | Pas de GPU actif ou pilote absent : le calcul se fait sur CPU. Vérifiez `!nvidia-smi` |
| `Auto-tests GPU … ÉCHEC : cette carte est ignorée` | Le noyau a donné un résultat inattendu sur cette carte. Le calcul continue sans elle ; ouvrez une *issue* avec la sortie complète |
| `Échec de la compilation CUDA` | Le script a basculé sur CPU. Lisez `build.log` dans le dossier de cache (`~/.cache/puzzle-gpu/<empreinte>/`) et joignez-le à une *issue* |
| `g++ est introuvable` | `sudo apt install g++` |
| `GPU inutilisable (… mémoire …)` | Filtre de Bloom trop gros pour la carte : réduisez la liste de cibles |
| Pas d'affichage animé dans le notebook | Normal : hors terminal, une ligne de statut est écrite toutes les 10 s (`-log S` pour changer) |
| Vitesse à 0 au début | Normal : la vitesse est lissée sur ~5 s |

## 10. Fonctionnement et limites

**Algorithme.** Pour chaque lot de 512 clés consécutives, un point central `kc·G` sert aux clés `kc−i` et `kc+i` ; une seule inversion modulaire (Montgomery, par lots) est partagée par tout le lot. Chaque clé est ensuite hachée (SHA-256 puis RIPEMD-160 pour les adresses) et testée dans un filtre de Bloom par blocs d'une ligne de cache ; seuls les candidats sont confirmés exactement sur CPU.

**GPU.** Un thread GPU traite un « travail » de plusieurs lots et démarre à n'importe quelle clé grâce à une table de multiplication à base fixe. Arithmétique 256 bits en mots de 32 bits. Un thread hôte par carte ; lancements auto-réglés (~0,4 s) pour garder Ctrl+C et l'affichage réactifs. Les cas dégénérés (plages minuscules, clés proches de 0) sont refaits sur CPU ; en cas d'erreur GPU, le calcul bascule sur CPU.

**Limites.**
- NVIDIA / CUDA uniquement ; P2PKH et P2WPKH de clés compressées uniquement.
- Le code GPU a été validé par émulation CPU et compilation `clang` (CUDA, sm_75), **pas encore mesuré sur toutes les cartes réelles** : la vitesse dépend de la carte et n'est pas garantie. Mesurez-la avec votre premier lancement et signalez tout problème via une *issue*.
- Doubler la plage double le temps de recherche : aucun mode ne contourne cette limite.

## Licence

À compléter avant publication (par exemple MIT). Sans licence, le code reste « tous droits réservés » par défaut.
