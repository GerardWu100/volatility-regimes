---
title: "Les régimes de volatilité sont faciles à trouver. Leur valeur prédictive l'est moins."
description: "Une étude sans fuite temporelle des régimes de surface de volatilité du SPX et du NDX, des caractéristiques en espace delta à l'embargo nécessaire pour tester les prévisions."
date: 2026-07-13
image: images/cover-volatility-regimes.png
categories: ["Quantitative Research", "Risk Management"]
---

# Les régimes de volatilité sont faciles à trouver. Leur valeur prédictive l'est moins.

Une surface d'options évolue rarement comme un seul chiffre. La volatilité implicite at-the-money peut monter pendant que le skew se creuse, que les ailes changent de forme et que les échéances courtes se repricent plus vite que les longues. Résumer tout l'épisode par « volatilité élevée » fait disparaître une bonne partie de l'information.

Ce projet pose une question plus précise : si je ramène chaque surface quotidienne du SPX et du NDX à un petit vecteur de caractéristiques, les états latents améliorent-ils la prévision de la volatilité réalisée sur les 20 prochaines séances ?

Le code comporte deux volets. Le pipeline descriptif repère des états sur l'échantillon complet. Le pipeline walk-forward réestime les modèles sur le passé, prédit un bloc à la fois et compare la prévision par régime à trois références : la volatilité implicite at-the-money courante, la volatilité réalisée historique et une régression linéaire. La distinction entre les deux volets est décisive. Un graphique de clusters bien net montre que la surface possède une structure. Il ne prouve pas que cette structure permet de prévoir la suite.

Le dépôt fournit des données de démonstration portables pour 3 912 séances, du 2010-01-04 au 2024-12-31. Les résultats ci-dessous décrivent cet échantillon suivi par Git. Ils ne remplacent pas une étude de production fondée sur un historique fournisseur vérifié séparément.

## Une surface quotidienne résumée en sept chiffres

Soit $\sigma_t(\Delta,\tau)$ la volatilité implicite annualisée à la date $t$, pour un delta signé $\Delta$ et une échéance $\tau$. Le delta fournit une coordonnée indépendante de l'échelle : une option 25-delta occupe une zone comparable de la surface même lorsque le niveau de l'indice change.

Pour chaque date, le constructeur choisit l'échéance la plus proche de 30 jours dans une plage de 15 à 45 jours, puis celle la plus proche de 90 jours dans une plage de 45 à 120 jours. Il interpole linéairement en delta et conserve sept valeurs :

$$
\begin{aligned}
\mathrm{ATM}_{t,\mathrm{near}} &= \sigma_t(-0.50,\tau_{\mathrm{near}}), \\
\mathrm{ATM}_{t,\mathrm{mid}} &= \sigma_t(-0.50,\tau_{\mathrm{mid}}), \\
\mathrm{Skew}_{t,\tau} &= \sigma_t(-0.25,\tau)-\sigma_t(+0.25,\tau), \\
\mathrm{Butterfly}_{t,\tau} &= \frac{\sigma_t(-0.25,\tau)+\sigma_t(+0.25,\tau)}{2}-\sigma_t(-0.50,\tau), \\
\mathrm{TermSlope}_t &= \mathrm{ATM}_{t,\mathrm{mid}}-\mathrm{ATM}_{t,\mathrm{near}}.
\end{aligned}
$$

Ici, $\mathrm{ATM}$ désigne la volatilité implicite at-the-money, $\mathrm{Skew}$ mesure la cherté relative de l'aile put par rapport à l'aile call, et $\mathrm{Butterfly}$ donne une mesure simple de la courbure. Les valeurs proches et intermédiaires de l'ATM, du skew et du butterfly fournissent six caractéristiques. La pente de terme constitue la septième.

L'interpolation reste volontairement prudente. Si le delta demandé se trouve hors de la plage observée, la fonction renvoie une valeur manquante. Elle n'extrapole pas une aile que les données ne montrent pas.

```python
delta_fraction = (target_delta - delta_low) / (delta_high - delta_low)
interpolated_iv = iv_low + (iv_high - iv_low) * delta_fraction

skew_value = put_wing_iv - call_wing_iv
butterfly_value = 0.5 * (put_wing_iv + call_wing_iv) - atm_iv
term_slope = atm_iv_mid - atm_iv_near
```

## Ce que fait le modèle d'états latents

Chaque colonne est standardisée avec sa moyenne et son écart-type dans l'échantillon. Soit $x_t$ le vecteur ainsi obtenu, de dimension sept. Un modèle de mélange gaussien à $K$ composantes lui attribue la densité suivante :

$$
p(x_t)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x_t\mid\mu_k,\Sigma_k),
$$

où $\pi_k$ est le poids de probabilité de la composante $k$, $\mu_k$ son vecteur moyen, $\Sigma_k$ sa matrice de covariance complète et $\mathcal{N}$ la densité normale multivariée.

Le code ajuste les valeurs de $K=2$ à $K=6$. Il retient le plus petit critère d'information bayésien, ou BIC :

$$
\mathrm{BIC}_K=-2\ell_K+p_K\log n,
$$

où $\ell_K$ est la log-vraisemblance ajustée, $p_K$ le nombre de paramètres estimés et $n$ le nombre d'observations quotidiennes. La pénalité empêche la vraisemblance de récompenser indéfiniment l'ajout d'états.

Les étiquettes d'un mélange n'ont pas d'ordre naturel. Le projet les trie donc selon leur volatilité implicite ATM proche moyenne. Le régime 0 correspond au niveau implicite le plus faible. Les numéros suivants correspondent à des niveaux moyens de plus en plus élevés. Ce tri facilite la lecture, mais il ne transforme pas un cluster non supervisé en prévision de risque.

![Volatilité implicite ATM proche du SPX avec les régimes descriptifs ajustés sur l'échantillon complet](images/01_spx_regime_timeline.png)

L'ajustement SPX sur l'échantillon complet retient cinq composantes. Les couleurs suivent les grands cycles de niveau des données portables, et les étiquettes séparent presque mécaniquement la volatilité implicite ATM. C'est logique : l'ATM proche est la première entrée du modèle et la variable utilisée pour ordonner les régimes.

## La cible de prévision, étape par étape

Soit $P_t$ le cours de clôture de l'indice à la séance $t$. On calcule d'abord le rendement logarithmique quotidien :

$$
r_t=\log\left(\frac{P_t}{P_{t-1}}\right).
$$

Pour un horizon de $h$ séances, on rassemble les rendements futurs $r_{t+1},\ldots,r_{t+h}$ et on calcule leur écart-type d'échantillon :

$$
s_t(h)=\operatorname{std}\left(r_{t+1},\ldots,r_{t+h}\right).
$$

Soit enfin $A$ le nombre de séances retenu pour l'annualisation. Le projet prend $A=252$, d'où la volatilité réalisée future :

$$
\mathrm{RV}_t(h)=s_t(h)\sqrt{A}.
$$

L'horizon par défaut est $h=20$. Toutes les valeurs restent en décimal dans les calculs : $0.15$ correspond à une volatilité annualisée de 15 pour cent.

Pour un état ordonné prédit $z_t$, le modèle utilise comme prévision la cible moyenne de cet état dans l'échantillon d'entraînement. Soit $\mathcal{T}_t$ l'ensemble des dates d'entraînement disponibles avant la date de prévision $t$. On obtient :

$$
\widehat{\mathrm{RV}}^{\mathrm{regime}}_t
=
\frac{
\sum_{u\in\mathcal{T}_t}\mathbf{1}\{z_u=z_t\}\mathrm{RV}_u(h)
}{
\sum_{u\in\mathcal{T}_t}\mathbf{1}\{z_u=z_t\}
},
$$

où $\mathbf{1}\{\cdot\}$ vaut un lorsque la condition est vraie et zéro sinon. La règle reste simple à dessein. Si elle apporte un gain, celui-ci doit venir de l'appartenance au régime, pas d'une couche de prévision flexible cachée derrière le modèle d'états.

## L'embargo qui rend le découpage honnête

Un découpage chronologique peut encore contenir une fuite. La cible attachée à une date d'entraînement $t$ utilise les rendements jusqu'à $t+h$. Si la fenêtre de test commence avant la fin de cette cible, le modèle apprend à partir de rendements qui appartiennent déjà à la période de test.

Soit $i(d)$ la position entière de la date $d$ dans l'historique de prix trié, et $s$ la première date de test. Une étiquette d'entraînement n'est admissible que si

$$
i(t)+h<i(s).
$$

Le moteur supprime chaque ligne d'entraînement qui ne respecte pas cette inégalité. Avec une cible à 20 jours, on obtient un écart de 20 séances entre la dernière étiquette d'entraînement utilisable et la première observation de test.

```python
test_start_position = price_date_positions[test_index[0]]
safe_train_dates = []

for train_date in train_index:
    forward_window_end = price_date_positions[train_date] + horizon
    if forward_window_end < test_start_position:
        safe_train_dates.append(train_date)
```

Ce détail apporte davantage de crédibilité qu'un modèle d'états supplémentaire. Sans cet embargo, la fenêtre croissante paraît causale alors que ses étiquettes franchissent discrètement la frontière du test.

## Ce que montre l'échantillon portable

J'ai régénéré l'analyse descriptive à partir des fichiers Parquet suivis par Git avec `blog/generate_charts.py`. Le script reprend les sept définitions de caractéristiques, la standardisation, les mélanges gaussiens à covariance complète, les cinq initialisations, la graine aléatoire 42 et la sélection par BIC du package.

| Symbol | Selected K | Lowest-regime ATM IV | Highest-regime ATM IV | Lowest-regime forward RV | Highest-regime forward RV |
|:--|--:|--:|--:|--:|--:|
| SPX | 5 | 10.01% | 19.96% | 15.25% | 14.65% |
| NDX | 3 | 11.47% | 20.26% | 15.07% | 14.83% |

![Moyennes de volatilité implicite et réalisée future par régime descriptif ordonné](images/02_regime_profiles.png)

Les courbes bleues montent par construction : les régimes sont ordonnés selon la volatilité implicite ATM proche. Les courbes orange, elles, ne montent pas. Dans ces données de démonstration, la volatilité réalisée moyenne sur les 20 séances suivantes reste proche de 15 pour cent dans tous les états. Pour le SPX, elle est même légèrement plus faible dans l'état le plus élevé que dans le plus bas. Pour le NDX, elle est presque plate.

La prime de risque de variance correspond ici à la volatilité implicite ATM moins la volatilité réalisée future. Dans le régime SPX le plus bas, sa moyenne vaut −5.24 points de pourcentage. Dans le plus haut, elle atteint +5.31 points. L'essentiel de cet écart vient de la volatilité implicite, sans évolution comparable de la volatilité réalisée par la suite.

![Volatilité implicite ATM proche face à la volatilité réalisée sur les 20 séances suivantes](images/03_atm_vs_forward_rv.png)

Le nuage de points raconte la même histoire sans moyenne par groupe. La corrélation entre la volatilité implicite ATM proche courante et la volatilité réalisée sur les 20 séances suivantes vaut −0.05 pour le SPX et −0.04 pour le NDX. Ces chiffres décrivent les données portables. Je ne les utiliserais pas pour une position ou une limite de risque. Ils suffisent toutefois à montrer qu'un graphique de régimes ne répond pas, à lui seul, à la question prédictive.

## Pourquoi il n'y a pas de tableau de victoire hors échantillon

La configuration par défaut exige une fenêtre d'entraînement initiale de 3 880 lignes. Chaque symbole compte 3 912 lignes complètes de caractéristiques. La cible future à 20 jours élimine les 20 dernières lignes, tandis que la référence de volatilité réalisée historique exige 20 jours d'historique au début. Après ces deux alignements, il reste 3 872 lignes. C'est moins que la taille minimale d'entraînement, donc le run walk-forward configuré produit à juste titre un panel de prévisions vide.

Abaisser le seuil après avoir vu ce résultat reviendrait à fabriquer une conclusion pour l'article. Je n'ai pas modifié la configuration de recherche.

Lorsqu'il existe des prévisions, le module de reporting calcule la racine de l'erreur quadratique moyenne, ou RMSE, l'erreur absolue moyenne, ou MAE, et un score hors échantillon relatif à la volatilité implicite ATM. Soit $\mathrm{MSE}_{m}$ l'erreur quadratique moyenne du modèle $m$ et $\mathrm{MSE}_{\mathrm{ATM}}$ celle de la référence sur les mêmes dates. Le score relatif est :

$$
R^2_{\mathrm{OOS},m}=1-\frac{\mathrm{MSE}_{m}}{\mathrm{MSE}_{\mathrm{ATM}}}.
$$

Une valeur positive indique que le modèle $m$ réduit l'erreur quadratique par rapport à l'ATM courant. Zéro correspond à une égalité. Une valeur négative signifie que le modèle plus élaboré fait moins bien.

## L'expérience suivante

Le prochain run doit réserver assez d'observations intactes pour juger les modèles. Je fixerais la fenêtre initiale avant d'examiner les erreurs, je conserverais l'embargo de 20 jours et je publierais les résultats par blocs temporels contigus, en plus du score agrégé. Une méthode de régimes qui ne gagne que sur une portion lisse de la série portable ne constitue pas une preuve convaincante.

Je comparerais aussi les ensembles de caractéristiques déjà déclarés dans le package : ATM seul, ATM avec structure par terme, ATM avec skew, smile de l'échéance proche et vecteur complet à sept dimensions. Cette comparaison permet de vérifier si la forme de la surface apporte quelque chose au-delà du niveau de volatilité qui sert à ordonner les états.

L'ordre de recherche du projet est sain. Il extrait la surface, trouve des états lisibles, définit une cible future, retire les étiquettes qui se chevauchent, puis compare le résultat à des prévisions simples. L'échantillon suivi par Git permet d'étudier les deux premières étapes et d'exercer le code des suivantes. Il ne permet pas encore d'affirmer que les régimes latents améliorent la prévision de la volatilité réalisée.
