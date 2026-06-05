# Energy Efficiency of Buildings: A Statistical Modelling Study

**Course:** DSA 8305 -- Linear Models  
**Institution:** Strathmore Institute of Mathematical Sciences, MSc Data Science and Analytics  
**Lecturer:** Dr. Evans Otieno Omondi

| Student | Registration Number |
|---|---|
| Assumpta Mwikali | 134022 |
| Olive Mideva Muloma | 135792 |
| Trevor Anjeyo Vuhyah | 224038 |
| Valerie Jerono | 222331 |

---

## The Problem We Are Solving

Imagine you are an architect about to design a residential building. The drawings are blank. No walls have been placed, no windows chosen, no height decided. At this moment -- before a single brick is laid -- the energy consumption of that building for the next 50 years is being determined. The height you choose, the ratio of glass to solid wall, the compactness of the footprint: these decisions lock in decades of heating bills and carbon emissions.

This is the problem we tackle in this study. Using the **ENB2012 dataset** (Tsanas and Xifara, 2012), a collection of 768 building configurations simulated using the Ecotect building energy tool, we ask: **can statistical models predict how much energy a building will consume, given only its architectural geometry?** And further: can we explain *why* -- which features matter most, and in what way?

The answer requires every model class taught in DSA 8305, deployed in sequence, each motivated by what the previous model could not do. That progression is the spine of this study.

---

## Dataset

| Variable | Name | Description |
|---|---|---|
| X1 | Relative Compactness | Ratio of volume to surface area -- higher means more cube-like |
| X2 | Surface Area | Total external surface in m2 |
| X3 | Wall Area | Total wall surface in m2 |
| X4 | Roof Area | Roof surface in m2 |
| X5 | Overall Height | Building height: only two values exist -- 3.5 m (single-storey) or 7.0 m (double-storey) |
| X6 | Orientation | Cardinal direction (2=N, 3=E, 4=S, 5=W) -- a nominal categorical variable |
| X7 | Glazing Area | Proportion of floor area that is glazed (0 to 0.4) |
| X8 | Glazing Area Distribution | How glazing is spread across facades |
| **Y1** | **Heating Load** | Annual energy for heating in kWh/m2 -- first response variable |
| **Y2** | **Cooling Load** | Annual energy for cooling in kWh/m2 -- second response variable |

**768 observations. No missing values. Two continuous response variables.**

---

## Methodology: CRISP-DM

This study is structured around the **Cross-Industry Standard Process for Data Mining (CRISP-DM)**, a widely adopted iterative framework for data science projects. The framework forces discipline: you cannot jump to modelling without understanding the data, and you cannot understand the data without first understanding the business problem.

```
Business Understanding
        |
        v
Data Understanding  <-------+
        |                   |
        v                   |
Data Preparation            |
        |                   |
        v                   |
    Modelling               |
        |                   |
        v                   |
    Evaluation  ------------+
        |
        v
  Deployment / Conclusions
```

Each phase is documented in the notebook. The arrows pointing back represent the iterative nature of real data science: findings in evaluation often send you back to re-examine your data preparation or try a different model.

---

## Phase 1: Business Understanding

Three questions drive this analysis, nested from broad to precise:

1. **Regression:** Given a building's geometry, predict its heating and cooling loads (kWh/m2).
2. **Classification:** Can buildings be labelled high-efficiency or low-efficiency before construction?
3. **Diagnostic:** Which features matter most, and is the relationship linear, curved, or interactive?

These questions map directly onto the model families in DSA 8305: regression models for questions 1 and 3, classification models for question 2.

---

## Phase 2: Data Understanding

### What the distributions tell us

Before fitting any model, we examine the shape of the response variables. This is not optional housekeeping -- it is one of the most informative steps in the entire analysis.

![Target Variable Distributions and Normality Assessment](images/Target%20Variable%20Distributions%20and%20Normality%20Assessment.png)


Both Heating Load (Y1) and Cooling Load (Y2) show **bimodal distributions** -- two distinct humps rather than a single bell curve. The Shapiro-Wilk test formally rejects normality for both (p < 10^-20).

Why two humps? The answer lies in X5 (Overall Height), which takes only two values in the entire dataset: 3.5 m for single-storey buildings and 7.0 m for double-storey buildings. Taller buildings require substantially more energy. The two height groups create two energy clusters, and those clusters produce the two humps we see in Y1 and Y2.

This finding has immediate implications. The normality assumption underpinning OLS t-tests is already under pressure. We cannot trust individual p-values blindly -- we must inspect residuals after fitting. And for classification, the two humps mean our high/low split is physically meaningful, not arbitrary.

### Correlations and multicollinearity

![Feature Correlations with Target Variables](images/Feature%20Correlations%20with%20Target%20Variables.png)


The correlation matrix reveals a critical structural problem. X1 (Relative Compactness) and X2 (Surface Area) have a Pearson correlation of approximately -0.99 -- near-perfect negative correlation. X4 (Roof Area) is strongly correlated with both. This is not a data quality problem. It is a geometric reality: a more compact building occupies less surface area per unit of volume, and a building's roof area scales with its footprint, which is constrained by its compactness. The physics of building geometry creates the correlation.

**Why does this matter for modelling?** In Ordinary Least Squares, the estimator is $\hat{\beta} = (X'X)^{-1}X'y$. When predictors are perfectly correlated, the matrix $X'X$ becomes singular -- it cannot be inverted, and the estimator breaks down. Near-perfect correlation inflates the standard errors of the correlated coefficients, making individual t-tests unreliable even though overall model predictions may still be reasonable. This is the multicollinearity problem, and it is the direct motivation for Ridge Regression and Principal Components Regression later in the analysis.

X5 (Overall Height) stands out as the strongest individual predictor of both Y1 and Y2. X6 (Orientation) is near-zero for both -- the simulation was balanced across all four cardinal directions, so any orientation effect averages out at the dataset level.

### The shape of each relationship

![Feature vs Heating Load](images/Feature%20vs%20Heating%20Load.png)

Each panel shows one feature plotted against Heating Load, with points coloured by building height (blue = 3.5 m, orange = 7.0 m). The black curve is a **LOWESS smoother** (Locally Weighted Scatterplot Smoothing).

LOWESS is worth explaining carefully because it will recur throughout the analysis. It is a **non-parametric smoother**: it makes no assumptions about functional form. At each point along the x-axis, it fits a local weighted regression using the neighbouring points, giving more weight to closer observations. The result is a curve that follows the data without committing to any particular equation. If the true relationship is linear, LOWESS traces a straight line. If it curves, LOWESS reveals the curve. This makes it an ideal diagnostic tool -- it shows us what the data actually say before we impose any model.

Several relationships stand out. X7 (Glazing Area) shows a clearly positive but non-linear association with Heating Load: more glazing means more heat loss, but the rate of increase is not constant. The near-flat LOWESS for X6 (Orientation) confirms the correlation finding. And crucially, the two coloured sub-clouds (blue and orange) follow different slopes across almost every feature -- evidence of **interaction effects** between Overall Height and the other features. A purely additive model will miss this structure.

### Quantifying the multicollinearity: VIF

![VIF by Feature](images/VIF%20by%20Feature.png)

The **Variance Inflation Factor** gives the correlation story a precise number. For predictor $j$, VIF is defined as:

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

where $R^2_j$ is the R-squared from regressing $X_j$ on all other predictors. A VIF of 10 means the standard error of that coefficient is $\sqrt{10} \approx 3.16$ times larger than it would be in a dataset without collinearity. VIF of infinity (shown in the table) means perfect linear dependence -- the variable is an exact linear combination of others. Surface Area, Wall Area, and Roof Area all return infinite VIF values, reflecting the geometric constraints of the ENB2012 simulation design. Relative Compactness returns VIF = 105.5. Overall Height, at VIF = 31.2, also has a concerning value. Only Glazing Area and Glazing Area Distribution are in the acceptable range.

The practical message: OLS will still generate reasonable predictions (collinearity does not bias predictions, only inflates their uncertainty), but individual coefficient tests are not trustworthy for the correlated features. Ridge and PCR are the prescribed remedies.

---

## Phase 3: Data Preparation

### One-hot encoding Orientation

Orientation takes values 2, 3, 4, 5 -- codes for North, East, South, West. Treating these as continuous numbers would imply North and South are twice as far apart as North and East, which is physically meaningless. We one-hot encode the variable, creating three binary indicator columns (one direction is dropped as the reference category to avoid the **dummy variable trap**).

The dummy variable trap is worth understanding. If we include all four direction dummies in a regression, they sum to 1 in every row -- perfect collinearity with the model's intercept column. The software cannot distinguish the intercept's effect from the combined effect of the four directions. Dropping one dummy (making it the reference) resolves this: each remaining coefficient is interpreted as the effect of that direction *relative to* the dropped reference direction.

### Train-test split and standardisation

We split 80% / 20% for training and test, stratified by load quantile to preserve the bimodal distribution in both sets. Standardisation (zero mean, unit variance) is applied only for Ridge Regression, PCR, and GAM -- models where coefficient magnitudes are either penalised or compared across features. OLS is fit on original scales for coefficient interpretability.

---

## Phase 4: Modelling

The models are introduced in order of increasing flexibility. This is deliberate. Each model is motivated by what the previous one could not do, and the progression maps directly onto the DSA 8305 syllabus.

---

### Model 1: Ordinary Least Squares -- The Parametric Baseline

The classical general linear model:

$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$$

OLS minimises the residual sum of squares to produce $\hat{\boldsymbol{\beta}} = (X'X)^{-1}X'y$. Under the Gauss-Markov assumptions, this is the **Best Linear Unbiased Estimator (BLUE)** -- no other linear unbiased estimator has smaller variance.

OLS for Heating Load achieves R-squared = 0.914 on the training set. For Cooling Load, R-squared = 0.888. Both F-statistics are highly significant (p << 0.001), confirming the model as a whole is informative. But a high R-squared does not mean the model is well-specified. We must inspect the residuals.


![Cooling OLS Diagnostics](images/Cooling%20OLS%20Diagnostics.png)
![Heating OLS Diagnostics](images/Heating%20OLS%20Diagnostics.png)

**Reading the four diagnostic plots:**

The **Residuals vs Fitted** plot should show random scatter around the horizontal zero line. Any curve or funnel shape signals a problem -- either non-linearity that the model is missing, or heteroskedasticity (non-constant variance). A curved LOWESS line in this plot is a direct call to add polynomial or spline terms.

The **Normal Q-Q plot** places the residuals on a probability scale. If they are normally distributed, the points fall on the diagonal red line. Departures at the tails indicate heavy-tailed or skewed residuals, which undermines the validity of t-tests and confidence intervals.

The **Scale-Location plot** (square root of absolute standardised residuals against fitted values) tests for homoskedasticity. A flat LOWESS line is what we want. An upward slope means variance grows with the fitted value -- a heteroskedastic pattern.

The **Actual vs Predicted** plot shows how well predictions track reality. Points clustering tightly along the 45-degree line indicate a well-calibrated model; systematic deviations above or below it reveal bias in specific regions of the response.

The Durbin-Watson statistics (1.74 for Y1, 1.79 for Y2) are close to 2.0, suggesting no serious autocorrelation in residuals -- reassuring for a cross-sectional dataset.

The OLS diagnostics establish a baseline against which every subsequent model is compared. They also reveal the specific limitations -- non-normality and potential heteroskedasticity from the bimodal target -- that motivate the more flexible models to follow.

---

### Model 2: Ridge Regression -- Addressing Multicollinearity

Ridge Regression (Hoerl and Kennard, 1970) modifies the OLS objective by adding an L2 penalty on coefficient magnitudes:

$$\hat{\boldsymbol{\beta}}^{\text{ridge}} = \underset{\boldsymbol{\beta}}{\arg\min} \left\| \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right\|^2 + \lambda \left\| \boldsymbol{\beta} \right\|^2$$

The closed-form solution is $(X'X + \lambda I)^{-1}X'y$. The key insight is in the addition of $\lambda I$: even if $X'X$ is near-singular due to collinearity, adding $\lambda I$ guarantees invertibility. This is the direct mathematical fix for the multicollinearity problem.

The regularisation parameter $\lambda$ (alpha in scikit-learn) controls the strength of the penalty. When $\lambda = 0$, Ridge reduces to OLS. As $\lambda \to \infty$, all coefficients are shrunk to zero and the model predicts the mean of Y regardless of X. The CV-optimal $\lambda$ sits between these extremes, trading a small amount of bias for a meaningful reduction in variance.

![Ridge Coefficient Paths](images/Ridge%20Coefficient%20Paths.png)

The coefficient path plot is one of the most instructive visualisations in regularised regression. Each line tracks one predictor's coefficient as $\lambda$ increases from near-zero (left) to very large (right). At the left edge, coefficients take their OLS values -- large, unstable, and potentially sign-reversed due to collinearity. As $\lambda$ increases, the chaotic coefficients stabilise. The red dashed line marks the CV-optimal $\lambda$, where Ridge has resolved most of the instability without over-penalising genuinely important predictors.

Ridge achieves test RMSE = 2.72 for Heating Load (R2 = 0.924) and RMSE = 3.21 for Cooling Load (R2 = 0.888) -- marginally better than OLS on the test set, confirming that the collinearity was mildly degrading OLS prediction performance.

---

### Model 3: Principal Components Regression -- A Geometric Solution

PCR takes a geometrically different approach to the same multicollinearity problem. Rather than penalising coefficients, it first **transforms** the correlated predictors into a set of uncorrelated, orthogonal axes (principal components), then regresses the response on the first $k$ of them.

The principal components are the eigenvectors of $X'X$, ordered by the amount of variance they explain. PC1 captures the largest source of variation in the feature space, PC2 the next largest, and so on. Because eigenvectors of a symmetric matrix are orthogonal, the components are uncorrelated by construction -- multicollinearity is geometrically eliminated.

![PCA Scree Plot](images/PCA%20Scree%20Plot.png)

The scree plot shows each component's individual contribution to total variance. The cumulative variance plot shows how many components are needed to retain a given proportion of information. In our dataset, 6 components explain 96.3% of the variance in X -- the remaining components add little information and mostly capture noise from the one-hot encoded orientation dummies.

![PCR Model Comparison](images/PCR%20Model%20Comparison.png)

Cross-validation selects 6 optimal components. PCR achieves test RMSE = 3.27 for Y1 -- slightly *worse* than OLS (2.72). This is an honest and instructive result. PCR is not guaranteed to outperform OLS. When the discarded components happen to contain signal relevant to Y (not just variance in X), dropping them hurts prediction. The lesson: **variance in X and covariance with Y are different things**. PCA optimises for the former; prediction requires the latter. Ridge, which retains all components but shrinks them proportionally to their contribution, is better suited to this dataset.

---

### Model 4: Polynomial Regression -- Non-linear Parametric Extension

Polynomial regression extends OLS by including powers of predictors as additional terms:

$$Y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d + \varepsilon$$

The model is non-linear in $x$ -- it can represent curves, inflection points, and S-shapes. But it remains **linear in the parameters** $(\beta_0, \ldots, \beta_d)$, so it is estimated by OLS on the expanded feature matrix. This is the key reason polynomial regression belongs within the linear models framework: non-linear in features, linear in parameters.

The risk is **overfitting**. A degree-5 polynomial with 8 predictors and all interaction terms produces hundreds of features. Ridge regularisation is applied inside the pipeline to control this expansion.

![Polynomial Regression](images/Polynomial%20Regression.png)

CV-RMSE decreases steadily from degree 1 through degree 5. Degree 5 is optimal for both Y1 (RMSE = 1.60, R2 = 0.974) and Y2 (RMSE = 2.18, R2 = 0.949) -- a substantial improvement over OLS. The improvement confirms what the LOWESS plots suggested: non-linear relationships exist in this data, and they carry meaningful predictive signal.

---

### Model 5: Regression Splines -- Piecewise Polynomial Smoothing

**Splines** are piecewise polynomial functions joined smoothly at **knot** points. A natural cubic spline is cubic between knots and linear beyond the outermost knots, preventing the erratic extrapolation behaviour of high-degree global polynomials.

The spline can be written as a linear combination of B-spline basis functions $B_j(x)$:

$$s(x) = \sum_{j=1}^{K+4} \beta_j B_j(x)$$

This converts a non-parametric problem back into OLS on an expanded feature matrix -- connecting splines directly to the linear models framework. The `patsy` library provides the `cr()` function for natural cubic spline basis expansion in Python.

![Natural Cubic Splines](images/Natural%20Cubic%20Splines.png)

The three panels show natural cubic splines fit to Glazing Area vs Heating Load with increasing degrees of freedom. The left panel (4 df) is too rigid and misses the curvature in the data -- high bias, low variance. The middle panel (7 df) follows the signal without chasing noise -- balanced bias-variance. The right panel (12 df) begins to fit individual noise structures in the data -- low bias, high variance, and the beginning of overfitting.

This figure is one of the most pedagogically valuable in the entire study: it makes the bias-variance trade-off visible with your own eyes. Every modelling decision -- the degree of polynomial, the number of knot degrees of freedom, the Ridge lambda -- is a movement along this curve.

---

### Model 6: Generalised Additive Models -- Semi-parametric Modelling

GAMs (Hastie and Tibshirani, 1986) represent one of the most elegant ideas in modern statistical modelling. They generalise the linear model by replacing each linear term $\beta_j x_j$ with an **arbitrary smooth function** $s_j(x_j)$:

$$Y = \alpha + s_1(x_1) + s_2(x_2) + \cdots + s_p(x_p) + \varepsilon$$

Each $s_j$ is estimated non-parametrically using penalised spline bases. The model retains the **additivity** of the linear model -- effects sum rather than multiply -- which makes each component directly interpretable. The overall model structure (additive, one term per predictor) is parametric. The individual functions $s_j$ are non-parametric. This combination is why GAMs are called **semi-parametric**.

**Penalised splines and automatic smoothness selection:** Under the hood, `pygam` fits each $s_j$ using a spline basis with a penalty on the second derivative of the function. This penalty discourages rapid oscillation and controls smoothness automatically -- we do not need to manually choose degrees of freedom as we did for the splines in the previous section. The smoothness parameter is tuned by **Generalised Cross-Validation (GCV)**, printed in the model summary.

![GAM Partial Dependence Plots](images/GAM%20Partial%20Dependence%20Plots.png)

The partial dependence plots are among the most interpretable outputs in the entire study. Each panel shows the marginal effect of one predictor on Heating Load, holding all other predictors at their average values, with 95% confidence intervals shaded in blue.

A few observations are particularly instructive:

**X5 (Overall Height):** The smooth shows a near-vertical step -- the GAM approximates the discrete jump between 3.5 m and 7.0 m buildings through a continuous non-parametric lens. This is a beautiful example of a non-parametric method adapting to data structure that no parametric form could specify in advance.

**X7 (Glazing Area):** A positive, curved relationship. The first increments of glazing have the largest per-unit impact on heating demand; the effect flattens at high glazing proportions. This is exactly the non-linearity the LOWESS plots suggested, now formally captured.

**X1 (Relative Compactness):** A non-linear effect suggesting an optimal range of compactness. Both very elongated and very cube-like buildings carry higher heating loads.

**Orientation dummies (l(7), l(8), l(9)):** All three are statistically non-significant (p > 0.30), confirming that orientation does not matter for total load prediction in this balanced simulation dataset.

**GAM performance:** RMSE = 1.12 for Y1 (R2 = 0.987), RMSE = 1.71 for Y2 (R2 = 0.968) -- the best regression performance of any model in the study. The GAM has successfully captured the non-linearities that OLS, Ridge, and PCR all missed.

**A note on p-values in GAMs:** The `pygam` summary warns that p-values for estimated smooth terms behave correctly only when smoothing parameters are known in advance, not estimated from data. When smoothing parameters are estimated (as they are here via GCV), the p-values tend to be lower than they should be -- the tests reject the null too readily. This is a known limitation of the pseudo-likelihood approach used in `pygam`. The warning should not be ignored, but it does not invalidate the smooth estimates or the predictions -- only the formal hypothesis tests.

---

### Model 7: Classification -- Logistic Regression and Decision Trees

Not every stakeholder needs a precise kWh prediction. A concept-stage architect might simply ask: *will this design be a high-energy or low-energy building?* This is a classification problem.

**Logistic Regression** is the natural classification extension of the linear model. For a binary outcome $Y \in \{0, 1\}$, it models the log-odds:

$$\log\frac{P(Y=1)}{P(Y=0)} = \mathbf{x}^\top\boldsymbol{\beta}$$

Equivalently, the probability of the positive class is:

$$P(Y=1 | \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{x}^\top\boldsymbol{\beta}}}$$

Parameters are estimated by maximum likelihood. Logistic regression is a member of the **Generalised Linear Models** family: same structural form as linear regression, but with a Bernoulli response distribution and a logit link function. This connects it directly to the GLM framework in DSA 8305.

**A crucial methodological correction:** An earlier version of this analysis defined the binary target by splitting Heating Load at the global median (18.95 kWh/m2). The resulting classifier achieved 97% accuracy and 0.99 AUC -- seemingly excellent. But the permutation importance plot revealed that Overall Height alone accounted for virtually 100% of the predictive power. Every other feature contributed zero. This happened because tall buildings almost always fall above the global median, and short buildings below it. The classifier learned one rule -- "is this a tall building?" -- and had no reason to learn anything else.

The corrected approach defines high/low load **within each height group** using group-specific medians. Now the classifier must use compactness, glazing, wall area, and roof configuration to distinguish efficient from inefficient buildings of the *same height*. This is the genuine classification problem.

![Classification Models Comparison](images/Classification%20Models%20Comparison.png)
![Feature Importance -- Decision Tree](images/Feature%20Importance%20--%20Decision%20Tree.png)

With the corrected labels, logistic regression achieves 83% accuracy and the decision tree achieves 91% accuracy. These are lower than the original 97% -- and that is the honest result. Multiple features now contribute meaningfully: Glazing Area (coefficient 2.01), Wall Area (1.90), and Roof Area (-1.01) are the strongest logistic regression predictors, which aligns perfectly with the GAM partial dependence plots.

The decision tree's confusion matrix shows a small number of misclassifications in each direction. In building design terms, a false negative (predicting low-load when the building will actually be high-load) is the more costly error -- the mechanical system will be under-specified. The precision-recall trade-off can be adjusted by changing the classification threshold from the default 0.5 depending on the cost asymmetry.

---

## Phase 5: Model Evaluation

![Model Comparison Dashboard](images/Model%20Comparison%20Dashboard.png)

All regression models evaluated on the same held-out test set:

| Model | Y1 RMSE | Y1 R2 | Y2 RMSE | Y2 R2 |
|---|---|---|---|---|
| GAM | **1.12** | **0.987** | **1.71** | **0.968** |
| Polynomial (d=5) | 1.60 | 0.974 | 2.18 | 0.949 |
| Ridge | 2.72 | 0.924 | 3.21 | 0.888 |
| OLS | 2.72 | 0.924 | 3.21 | 0.888 |
| PCR | 3.27 | 0.890 | 3.88 | 0.837 |

**Reading the table:** RMSE is in kWh/m2 -- the same unit as the response variable, which makes it directly interpretable. The GAM's RMSE of 1.12 kWh/m2 for Heating Load means its average prediction error is about 1.12 kWh/m2. For context, Heating Load ranges from roughly 6 to 43 kWh/m2 -- so the GAM is typically within 3% of the true value.

**The model progression tells a coherent story.** OLS and Ridge perform identically on this dataset (the collinearity is not severe enough to meaningfully hurt OLS predictions, only its coefficient standard errors). PCR underperforms both -- it discards components that carry signal for Y even though they carry low variance in X. Polynomial regression with degree 5 makes a substantial jump, confirming the presence of non-linear effects. GAM makes a further jump, confirming that the smooth, adaptive functions capture structure that fixed-degree polynomials still miss.

The improvement from OLS (R2 = 0.924) to GAM (R2 = 0.987) for Heating Load represents a reduction in unexplained variance from 7.6% to 1.3% -- meaningful both statistically and practically.

---

## Phase 6: Conclusions

### What we learned about buildings

**Height is the primary energy lever.** A single-storey building will almost always have substantially lower heating and cooling loads than a geometrically equivalent double-storey design. For any architect with a hard energy target, this is the most impactful design decision.

**Compactness matters, but non-linearly.** The GAM smooth for Relative Compactness suggests an optimal range -- neither maximally compact nor maximally elongated minimises energy load. The optimal shape is a moderate rectangular footprint.

**Glazing requires careful management.** More glass increases heating demand. The relationship is non-linear: the first increments of glazing have the largest per-unit impact. Highly glazed buildings need compensating insulation.

**Orientation does not matter in this simulation.** The balanced design of ENB2012 averages out orientation effects. In real buildings at specific sites, orientation matters considerably for solar gain and natural ventilation.

### What we learned about modelling

**Multicollinearity inflates uncertainty but does not necessarily harm prediction.** Ridge and OLS predicted equally well, but for inference, Ridge's more stable coefficients are preferable when collinearity is severe.

**Non-linearity is real and consequential.** The jump from OLS (R2 = 0.924) to GAM (R2 = 0.987) represents a 64% reduction in unexplained variance. Ignoring non-linearity in this dataset would materially mislead any engineering decision.

**PCR is not a universal improvement.** Dimensionality reduction helps when discarded components are genuinely noisy. When they carry signal for Y (as here, where later components include orientation information and height interactions), PCR hurts prediction. Model selection must always be evaluated empirically on held-out data.

**Classification labels must be designed carefully.** The original global-median split produced a trivially easy problem that one variable solved. The corrected within-group split produced a genuinely multivariate classification problem with realistic accuracy levels. The lesson: always interrogate whether your target variable construction is doing what you think it is.

---

## Why Your EDA Plots Are Not Saving to the Repository

This is a common and important practical point. Throughout the notebook, figures are displayed using `plt.show()`. This renders the plot in the notebook's output cell, but it does **not** write any file to disk. GitHub (or any other repository) can only track files that exist on the filesystem. It cannot read image data that is embedded inside a `.ipynb` output cell.

To save plots as standalone image files that you can commit to a repository, add `plt.savefig()` **before** `plt.show()` in each plotting cell:

```python
plt.tight_layout()
plt.savefig('images/Target Variable Distributions and Normality Assessment.png', bbox_inches='tight', dpi=150)
plt.show()
```

Then create an `images/` folder in your repository, commit all the saved `.png` files, and reference them in your README using the relative path syntax:

```markdown
![Target Variable Distributions and Normality Assessment](images/Target%20Variable%20Distributions%20and%20Normality%20Assessment.png)

```

For this README, all plots were extracted directly from the notebook's embedded output data and saved to the `images/` folder. Going forward, add `savefig` calls to each plotting cell so the images persist independently of the notebook execution state.

---

## Repository Structure

```
.
├── README.md                              -- This file
├── Energy_Efficiency_Analysis_Improved.ipynb  -- Main analysis notebook
├── ENB2012_data 2.xlsx                    -- Dataset
└── images/
    ├── Target Variable Distributions and Normality Assessment.png
    ├── Feature Correlations with Target Variables.png
    ├── Feature vs Heating Load.png
    ├── VIF by Feature.png
    ├── Cooling OLS Diagnostics.png
    ├── Heating OLS Diagnostics.png
    ├── Ridge Coefficient Paths.png
    ├── PCA Scree Plot.png.png
    ├── PCR Model Comparison.png
    ├── Polynomial Regression.png
    ├── Natural Cubic Splines.png
    ├── GAM Partial Dependence Plots.png
    ├── Classification Models Comparison.png
    ├── Feature Importance -- Decision Tree.png
    └── Model Comparison Dashboard.png
```

---

## References

1. Tsanas, A., and Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. *Energy and Buildings*, 49, 560--567.
2. Hastie, T., and Tibshirani, R. (1986). Generalized additive models. *Statistical Science*, 1(3), 297--310.
3. Hoerl, A. E., and Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55--67.
4. Dobson, A. J., and Barnett, A. G. (2018). *An Introduction to Generalized Linear Models* (4th ed.). Chapman and Hall/CRC.
5. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
6. James, G., Witten, D., Hastie, T., and Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.



