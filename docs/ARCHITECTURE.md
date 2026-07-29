# Architecture

Ce document explique les décisions. Le détail de leur mise en œuvre est dans le
code, qui est commenté pour être lu : [`futo/model.py`](../futo/model.py).

## Le principe directeur

Rien de neuf. Futo assemble ce qui s'est stabilisé dans les décodeurs depuis
Llama, en écartant tout ce qui apporte de la complexité sans gain mesurable à
l'échelle visée (100 M à 1,2 G de paramètres). L'originalité du projet est dans
le tokenizer et les données, pas dans l'architecture — et prétendre le contraire
serait malhonnête.

## Le bloc

```
x ← x + Attention(RMSNorm(x))
x ← x + SwiGLU(RMSNorm(x))
```

**Pré-normalisation.** La norme s'applique *avant* le sous-module, et le
résiduel n'est pas normalisé. Le chemin résiduel reste ainsi propre de l'entrée
à la sortie, ce qui permet d'empiler des dizaines de couches sans réglage fin de
l'échauffement. La post-normalisation du transformeur original demande un
warmup soigneux et diverge plus facilement.

**RMSNorm** plutôt que LayerNorm : pas de moyenne à retrancher, pas de biais,
donc moins de calcul, pour une stabilité équivalente sur les décodeurs. La
statistique est calculée en float32 quel que soit le dtype d'entrée — la somme
des carrés perd trop de précision en bf16.

**RoPE** plutôt qu'ALiBi ou des positions apprises. Les positions rotatives
encodent la position dans la *phase* des vecteurs requête et clé, de sorte que
le produit scalaire entre deux positions ne dépend que de leur écart. C'est
vérifié par un test (`test_rope_ne_depend_que_de_la_distance`). Les tables
cos/sin sont calculées en float32 : en bf16, `1 / 10000^(2i/d)` perd assez de
précision pour que deux positions éloignées finissent avec le même angle.

**GQA** (attention à requêtes groupées) : plusieurs têtes de requête partagent
une tête clé/valeur. À qualité quasi identique, le cache KV de l'inférence est
divisé par `n_head / n_kv_head`. C'est ce qui rend la génération tenable en
mémoire, et c'est gratuit à l'entraînement.

**SwiGLU** : `down(silu(gate(x)) · up(x))`. Trois matrices au lieu de deux, mais
une dimension cachée réduite d'un tiers (8/3 · d_model au lieu de 4 · d_model)
pour conserver le même nombre de paramètres. Gain net constaté sur la
perplexité à budget égal.

**Aucun biais**, nulle part. Ils n'apportent rien de mesurable dans un décodeur
pré-normalisé et compliquent la répartition du weight decay.

**Poids liés** entre l'embedding d'entrée et la tête de sortie. Sur `futo-small`,
cela économise 25,2 M de paramètres sur 126 M — un quart du modèle. À cette
échelle, l'économie vaut largement la légère perte de capacité.

## Ce qui est écarté, et pourquoi

| Écarté | Raison |
|---|---|
| Mélange d'experts (MoE) | Complexité d'entraînement et de service sans rapport avec la taille visée. À reconsidérer au-delà de 7 G de paramètres. |
| ALiBi | RoPE fait mieux et s'est imposé. |
| Attention linéaire, Mamba | Le gain n'apparaît qu'aux contextes très longs ; à 1 024-2 048 tokens, l'attention quadratique ne coûte presque rien. |
| MQA (une seule tête kv) | Dégradation mesurable ; GQA offre le même gain mémoire pour une perte moindre. |
| Logit soft-capping | Utile sur de très gros modèles instables ; inutile ici et coûteux en calcul. |
| QK-norm | Stabilise les entraînements très grands ; complexité non justifiée sous 1 G de paramètres. |

## Initialisation

Tout est tiré d'une loi normale d'écart-type 0,02, **sauf** les projections qui
écrivent dans le chemin résiduel — `o_proj` de l'attention et `down_proj` du
MLP — dont l'écart-type est divisé par `√(2·n_layer)`.

Sans cette correction, la variance des activations croît couche après couche et
les premiers pas divergent. C'est le piège n° 4 documenté dans le code.

## Le weight decay

Il ne s'applique qu'aux tenseurs de dimension ≥ 2 : matrices de projection et
embeddings. Les gains de RMSNorm sont de dimension 1 ; les régulariser les tire
vers zéro et éteint progressivement les couches. Un test vérifie que la
répartition est exhaustive et sans doublon
(`test_groupes_de_parametres_excluent_les_normes`).

## Le masque d'attention

C'est le point le plus délicat du fichier. `scaled_dot_product_attention` aligne
son masque causal **en haut à gauche**, ce qui rend `is_causal=True` correct
dans un seul des trois cas d'usage :

| Situation | `q_len` vs `kv_len` | Masque correct |
|---|---|---|
| Entraînement, préremplissage complet | égaux | `is_causal=True` |
| Décodage token par token | `q_len = 1` | **aucun masque** — la requête voit tout le cache |
| Préremplissage par morceaux | `1 < q_len < kv_len` | masque explicitement **décalé** |

Employer `is_causal=True` dans le deuxième cas masquerait tout sauf le premier
token. Le modèle s'entraînerait parfaitement — l'entraînement, lui, est bien
dans le premier cas — et ne générerait que du bruit. C'est exactement le genre
de faute qu'on ne découvre qu'après avoir payé le GPU, d'où le test
`test_cache_kv_equivaut_a_la_passe_complete`.

## Le calcul des FLOPs

Utilisé pour le MFU et pour les estimations de durée de `futo info` :

```
FLOPs/token = 6 · N_denses + 12 · n_layer · d_model · block_size
```

Le premier terme suit la convention de Kaplan : une passe avant coûte ≈ 2·N
opérations par token, l'arrière le double. `N_denses` compte les projections des
couches **et la tête de sortie** (qui fait bien une multiplication matricielle),
mais **pas** la table d'embedding, qui n'est qu'une consultation.

Le second terme est le coût de l'attention elle-même — quadratique en la
longueur de contexte, et totalement absent du comptage de paramètres. L'omettre,
comme le fait la règle simplifiée « 6ND », surestime le MFU des modèles à long
contexte. Un test vérifie que doubler le contexte augmente bien le résultat
(`test_flops_incluent_le_cout_de_lattention`).

C'est ce qui explique l'écart avec un calcul de coin de table : sur
`futo-small`, 6ND donne 6,0 EFLOP là où le compte complet en donne 7,1.

## Les quatre tailles

| | `tiny` | `small` | `base` | `large` |
|---|---|---|---|---|
| Paramètres | 1,3 M | 100,7 M | 299,4 M | 1 180,8 M |
| Couches | 4 | 12 | 24 | 24 |
| `d_model` | 128 | 768 | 1 024 | 2 048 |
| Têtes (requête / kv) | 4 / 2 | 12 / 4 | 16 / 4 | 16 / 8 |
| `d_ff` | 384 | 2 048 | 2 752 | 5 504 |
| Contexte | 256 | 1 024 | 2 048 | 2 048 |
| Vocabulaire | 4 096 | 32 768 | 32 768 | 32 768 |

Le vocabulaire est fixé à 32 768 pour les trois tailles utiles. Deux raisons :
il tient dans un `uint16`, ce qui divise par deux le poids des shards sur disque
(20 Go au lieu de 40 pour 10 G tokens) ; et c'est une puissance de deux, ce qui
convient aux noyaux matriciels des GPU.

La formule de comptage des paramètres est vérifiée contre le modèle réel, au
paramètre près, sur plusieurs configurations
(`test_comptage_parametres_analytique`). C'est ce qui permet de dimensionner une
location de GPU sans instancier le modèle.
