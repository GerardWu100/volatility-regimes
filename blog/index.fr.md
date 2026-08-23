---
title: "Les régimes de volatilité sont faciles à trouver. Leur valeur prédictive l'est moins."
description: "Une étude sans fuite temporelle des régimes de surface de volatilité du SPX et du NDX, des caractéristiques en espace delta à l'embargo nécessaire pour tester les prévisions."
date: 2026-07-13
image: images/cover-volatility-regimes.png
categories: ["Quantitative Research", "Risk Management"]
---

# Les régimes de volatilité sont faciles à trouver. Leur valeur prédictive l'est moins.

Une surface d'options évolue rarement comme un seul chiffre. La volatilité implicite at-the-money peut monter pendant que le skew se creuse, que les ailes changent de forme et que les échéances courtes se repricent plus vite que les longues. Résumer tout l'épisode par "volatilité élevée" fait disparaître une bonne partie de l'information.

Je voulais répondre à une question plus précise. Si je ramène chaque surface quotidienne du SPX et du NDX à un petit vecteur de caractéristiques, les états latents améliorent-ils la prévision de la volatilité réalisée sur les 20 prochaines séances ?

Le code comporte deux volets. Le pipeline descriptif repère des états sur l'échantillon complet. Le pipeline walk-forward réestime les modèles sur le passé, prédit un bloc à la fois et compare la prévision par régime à la volatilité implicite at-the-money courante, à la moyenne historique croissante, à la volatilité réalisée historique et à une régression linéaire. Je sépare ces tâches pour une raison précise. Un graphique de clusters net montre une structure dans la surface. Il ne dit encore rien de sa valeur prédictive.

Le dépôt fournit des données de démonstration portables pour 3,912 séances, du 2010-01-04 au 2024-12-31. Les résultats ci-dessous décrivent cet échantillon suivi par Git. Ils ne remplacent pas une étude de production fondée sur un historique fournisseur vérifié séparément.

## Une surface quotidienne résumée en sept chiffres

Soit $\sigma_t(\Delta,\tau)$ la volatilité implicite annualisée à la date $t$, pour un delta signé $\Delta$ et une échéance $\tau$. Le delta fournit une coordonnée indépendante de l'échelle. Une option 25-delta occupe une zone comparable de la surface même lorsque le niveau de l'indice change.

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

Chaque colonne est standardisée avec sa moyenne et son écart-type. Le graphique descriptif utilise les estimations de l'échantillon complet. Chaque ajustement walk-forward estime son échelle sur la seule fenêtre d'entraînement. Soit $x_t$ le vecteur de dimension $d$ ainsi obtenu. Le modèle descriptif complet prend $d=7$, tandis que la prévision par défaut utilise $d=3$ : ATM proche, ATM intermédiaire et pente de terme. Un modèle de mélange gaussien, ou GMM, à $K$ composantes lui attribue la densité suivante :

$$
p(x_t)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x_t\mid\mu_k,\Sigma_k),
$$

où $\pi_k$ est le poids de probabilité de la composante $k$, $\mu_k$ son vecteur moyen, $\Sigma_k$ sa matrice de covariance complète et $\mathcal{N}$ la densité normale multivariée.

Le code ajuste les valeurs de $K=2$ à $K=6$. Il retient le plus petit critère d'information bayésien, ou BIC :

$$
\mathrm{BIC}_K=-2\ell_K+p_K\log n,
$$

où $\ell_K$ est la log-vraisemblance ajustée, $p_K$ le nombre de paramètres estimés et $n$ le nombre d'observations quotidiennes. Avec des matrices de covariance complètes en dimension $d$, le nombre de paramètres vaut

$$
p_K=(K-1)+Kd+K\frac{d(d+1)}{2}.
$$

Les trois termes comptent les poids indépendants du mélange, les moyennes des composantes et les éléments distincts des matrices de covariance. La pénalité empêche la vraisemblance de récompenser indéfiniment l'ajout d'états. Le pipeline descriptif considère $K=2,\ldots,6$. La configuration walk-forward considère $K\in\{2,3\}$ dans chaque fenêtre d'entraînement.

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

Ce détail compte davantage que l'ajout d'un modèle d'états. Sans cet embargo, la fenêtre croissante paraît causale alors que ses étiquettes franchissent la frontière du test.

## Réparer une expérience qui ne pouvait pas démarrer

La configuration initiale demandait 3,880 lignes d'entraînement. Chaque symbole compte 3,912 lignes complètes de caractéristiques. La cible future à 20 jours supprime les 20 dernières lignes, tandis que la référence de volatilité réalisée historique exige 20 rendements passés au début. Le panel commun ne contient donc plus que 3,872 lignes, soit moins que la fenêtre demandée avant même d'appliquer l'embargo.

Écrire silencieusement des fichiers CSV vides n'était pas le bon comportement. Le contrat corrigé définit `min_train_size` comme le nombre de lignes étiquetées qui subsistent après l'embargo. Sa nouvelle valeur est 2,520, soit environ dix années de bourse. Le moteur ignore les premiers découpages candidats jusqu'à disposer de 2,520 étiquettes admissibles. Un contrôle préalable calcule aussi l'historique maximal qui peut laisser une observation de test. Une demande impossible déclenche maintenant une erreur qui indique le nombre de lignes alignées, le maximum admissible et le minimum demandé.

Les cinq dates d'un bloc de test partagent une fenêtre d'entraînement fixe. Les modèles linéaire et GMM sont maintenant ajustés une fois pour le bloc, puis évaluent les cinq lignes. Ce changement supprime des calculs répétés sans modifier l'ensemble d'information ni les prévisions. La première prévision valide tombe le 2019-10-28. La dernière tombe le 2024-12-03, car les dates restantes de décembre ne disposent pas encore d'une cible future complète à 20 jours.

## Mesurer la perte de prévision

Soit $y_t$ la volatilité réalisée et $\hat y_{m,t}$ la prévision du modèle $m$ à la date $t$. Pour $N$ dates de prévision, l'erreur est $e_{m,t}=y_t-\hat y_{m,t}$. L'erreur quadratique moyenne, ou MSE, sa racine, ou RMSE, et l'erreur absolue moyenne, ou MAE, sont

$$
\begin{aligned}
\mathrm{MSE}_m &= \frac{1}{N}\sum_{t=1}^{N}e_{m,t}^2, \\
\mathrm{RMSE}_m &= \sqrt{\mathrm{MSE}_m}, \\
\mathrm{MAE}_m &= \frac{1}{N}\sum_{t=1}^{N}|e_{m,t}|.
\end{aligned}
$$

La RMSE et la MAE ont la même unité décimale annualisée que la cible. Le graphique les multiplie par 100 : 0.024924 devient 2.4924 points de pourcentage de volatilité.

Pour une référence $b$, le score relatif hors échantillon est

$$
R^2_{\mathrm{OOS},m\mid b}=1-\frac{\mathrm{MSE}_m}{\mathrm{MSE}_b}.
$$

Une valeur positive indique que le modèle $m$ produit une erreur quadratique plus faible que la référence $b$ sur les mêmes dates. Le rapport calcule ce score par rapport à l'ATM implicite courant et à la moyenne historique croissante. La seconde comparaison est la plus exigeante : elle vérifie si le modèle apporte de l'information au-delà du niveau inconditionnel de la cible observé jusque-là.

## Le résultat corrigé

Le run par défaut produit 1,332 prévisions par symbole à un horizon de 20 jours. Chaque fenêtre d'entraînement choisit $K$ par BIC dans $\{2,3\}$. Le SPX utilise deux états sur 1,022 dates de prévision et trois sur 310. Le NDX en utilise deux sur 1,012 dates et trois sur 320.

| Symbol | Model | RMSE (pp) | MAE (pp) | $R^2_{\mathrm{OOS}}$ vs historical mean |
|:--|:--|--:|--:|--:|
| SPX | Historical mean | 2.4924 | 2.0243 | 0.0000 |
| SPX | GMM regime mean | 2.4932 | 2.0305 | -0.0007 |
| SPX | Linear features | 2.4999 | 2.0342 | -0.0061 |
| SPX | Trailing realized volatility | 3.6150 | 2.8981 | -1.1038 |
| SPX | Current ATM IV | 4.7103 | 3.9488 | -2.5717 |
| NDX | Historical mean | 2.5297 | 2.0374 | 0.0000 |
| NDX | GMM regime mean | 2.5442 | 2.0491 | -0.0115 |
| NDX | Linear features | 2.5355 | 2.0336 | -0.0046 |
| NDX | Trailing realized volatility | 3.8561 | 3.0798 | -1.3235 |
| NDX | Current ATM IV | 5.2400 | 4.3312 | -3.2907 |

![Out-of-sample RMSE for five realized-volatility forecasts](images/02_oos_rmse.png)

La moyenne par régime réduit fortement la RMSE par rapport à l'ATM implicite courant, mais la moyenne historique croissante fait légèrement mieux pour les deux symboles. L'écart de RMSE n'est que de 0.0008 point de pourcentage pour le SPX. Il atteint 0.0145 point pour le NDX. La régression linéaire reste elle aussi très proche. Dans cet échantillon, la structure de surface n'améliore pas la prévision quadratique agrégée par rapport à une moyenne inconditionnelle lentement révisée.

Cette comparaison change aussi la lecture du résultat ATM. La volatilité implicite est un prix sous probabilité risque-neutre qui contient une prime de risque de variance, tandis que la volatilité réalisée est un résultat sous la mesure physique. La poche d'options proche vise environ 30 jours calendaires et la cible couvre 20 séances. Une forte erreur de l'ATM ne prouve pas une erreur de prix exploitable. Le battre n'isole pas non plus une information propre aux régimes.

![Cumulative GMM squared-error loss minus historical-mean loss](images/03_cumulative_loss_difference.png)

La différence de perte cumulée utilise les erreurs quadratiques en points de pourcentage. Une courbe descendante favorise le GMM. Une courbe montante favorise la moyenne historique. Le SPX gagne nettement pendant une partie de 2020 et de 2021, puis rend cet avantage et termine à +5.68 points de pourcentage au carré. Le NDX termine à +98.01. Le chemin est instable, même lorsque l'écart final du SPX paraît minuscule.

## Ce que le résultat établit, et ses limites

L'hypothèse centrale échoue sur les données de démonstration avec cette configuration. Les régimes GMM décrivent nettement la surface d'options, mais leurs moyennes conditionnelles ne battent pas la moyenne historique croissante entre le 2019-10-28 et le 2024-12-03.

Cette conclusion reste étroite. Les fichiers Parquet suivis par Git sont des données pédagogiques, pas un historique fournisseur vérifié indépendamment. Le run ne teste qu'un horizon de 20 jours et l'ensemble de trois colonnes `atm_term`. Les cibles adjacentes partagent 19 rendements sur 20. Les 1,332 erreurs quotidiennes ne constituent donc pas 1,332 observations indépendantes. Le rapport fournit des estimations ponctuelles sans inférence corrigée du chevauchement. La sélection du nombre d'états par BIC et les paramètres sont réestimés toutes les cinq dates. Le résultat peut donc dépendre du calendrier de réestimation, et le calcul reste coûteux.

Une étude de production devrait figer son protocole avant d'observer les pertes, tester les ensembles de caractéristiques déclarés et plusieurs horizons, puis publier des résultats sur blocs non chevauchants ou une incertitude corrigée de l'hétéroscédasticité et de l'autocorrélation. Elle devrait aussi vérifier le calibrage sur de vraies cotations d'options : qualité des cotations, conventions de delta, interpolation des échéances et distinction entre volatilité implicite et physique.

## Références primaires

- Dempster, Laird, and Rubin (1977), [“Maximum Likelihood from Incomplete Data via the EM Algorithm”](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x), pour l'estimation par espérance-maximisation.
- Schwarz (1978), [“Estimating the Dimension of a Model”](https://doi.org/10.1214/aos/1176344136), pour le critère d'information bayésien.
- Andersen, Bollerslev, Diebold, and Labys (2003), [“Modeling and Forecasting Realized Volatility”](https://doi.org/10.1111/1468-0262.00418), pour la mesure et la prévision de la volatilité réalisée.
- Campbell and Thompson (2008), [“Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?”](https://doi.org/10.1093/rfs/hhm055), pour la moyenne historique comme référence et le score relatif $R^2$ hors échantillon.

L'échec de la comparaison est le résultat. Dès que la moyenne inconditionnelle entre dans le panel, l'avantage apparent des régimes disparaît.
