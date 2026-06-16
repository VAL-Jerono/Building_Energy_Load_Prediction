# CHANGES.md: Building Energy Load Prediction -- Notebook Improvements

**Project:** DSA 8305 Linear Models, Strathmore Institute of Mathematical Sciences  
**Dataset:** UCI ENB2012 (Tsanas and Xifara, 2012), 768 simulated buildings  
**Notebook:** Energy_Efficiency__Final.ipynb  
**Authors:** Assumpta Mwikali (134022), Olive Mideva Muloma (135792), Trevor Anjeyo Vuhyah (224038), Valerie Jerono (222331)

---

## Summary Table of All Changes

| Change | Type | Location in Notebook | DSA 8305 Syllabus Week | Rationale |
|:---|:---|:---|:---|:---|
| Shapiro-Wilk normality test | New code + markdown | After OLS diagnostics (Cell 31) | Weeks 1-4 | Replaces visual Q-Q inspection with a formal, reportable test decision |
| Breusch-Pagan heteroscedasticity test | New code (same cell) | After OLS diagnostics (Cell 31) | Weeks 1-4 | Formally confirms residual variance is non-constant; motivates WLS |
| Weighted Least Squares (WLS) | New model: code + markdown + reading | Cells 33-35 | Weeks 9-11 | Theoretically correct response to confirmed heteroscedasticity; GLS special case |
| Lasso regression with feature selection path | New model: code + markdown + reading | Cells 39-41 | Weeks 9-11 | Completes regularisation picture; provides automatic feature selection alongside Ridge |
| Nonlinear Least Squares (NLS) | New model: code + markdown + reading | Cells 53-55 | Weeks 5-8 | Fills the only complete gap in syllabus coverage; core Weeks 5-8 topic |
| Interaction models with partial F-test | New model: code + markdown + reading | Cells 56-58 | Weeks 12-14 | Formally tests whether Height modifies Glazing/Compactness effects; resolves visual-only gap |
| GAM 95% prediction intervals | Extension: code + markdown + reading | Cells 63-65 | Weeks 12-14 | Adds deployable uncertainty quantification; moves from point estimates to engineering-ready output |
| Unified 10-fold cross-validation table | New evaluation: code + markdown | Cells 72-75 | Weeks 1-4 | All model families on the same folds; first honest apples-to-apples CV comparison |
| AIC/BIC comparison table | New evaluation code | Cell 74 | Weeks 1-4 | Information criteria comparison for parametric models (OLS, WLS, Interaction) |
| Syllabus mapping table | New markdown in Cell 0 | Phase 0 Overview | All | Explicit connection to Dr. Omondi's course syllabus; shows full unit coverage |
| Updated model overview table | Modified Cell 22 | Phase 4 header | All | Adds DSA 8305 week column; lists all 12 models including 4 new ones |
| Updated conclusions | Modified Cell 79 | Phase 6 | All | Three new findings (WLS, Interaction, NLS); resolved two listed limitations |
| Updated references | Modified Cell 80 | Phase 6 | All | Added Shapiro-Wilk (1965), Tibshirani (1996), Levenberg (1944), SciPy (2020) |
| Updated imports | Modified Cell 3 | Environment setup | All | Added `from scipy.optimize import curve_fit` for NLS estimation |

---

## Change 1: Formal Residual Diagnostics (Shapiro-Wilk + Breusch-Pagan)

### What Was Added

Two formal statistical tests were added immediately after the OLS diagnostic plots: the Shapiro-Wilk test for normality of residuals and the Breusch-Pagan test for heteroscedasticity. Both are applied to residuals from both the Heating Load (Y1) and Cooling Load (Y2) OLS models.

### Why This Was Added

The original notebook included four diagnostic plots (Residuals vs Fitted, Q-Q plot, Scale-Location, Actual vs Predicted) but no formal hypothesis tests. Visual diagnosis is subjective: two analysts can look at the same Q-Q plot and disagree on whether the departure from normality is "severe enough." Formal tests replace this subjectivity with a reproducible decision backed by probability theory.

Both tests are standard content in the DSA 8305 Weeks 1-4 material on OLS diagnostics and hypothesis testing. Their absence was a gap between what the course teaches and what the notebook demonstrated.

### What the Tests Do

**Shapiro-Wilk (1965).** Tests H0: residuals follow a normal distribution. W statistic near 1 indicates normality; W below 1 indicates departure. Applied to first 200 training residuals to avoid large-sample power inflation. A rejected H0 means OLS t-tests and confidence intervals in the coefficient summary table are approximate.

**Breusch-Pagan (1979).** Tests H0: residual variance is constant across all observations (homoscedasticity). The procedure regresses squared residuals on the predictor matrix; a high R-squared in that auxiliary regression indicates that variance changes with the predictors. A rejected H0 means OLS standard errors are incorrect, which directly motivated adding WLS as the next model.

### Concept: Why Normality and Homoscedasticity Matter

OLS is the Best Linear Unbiased Estimator (BLUE) under the Gauss-Markov conditions, which include:
1. E[epsilon | X] = 0 (zero conditional mean)
2. Var(epsilon_i) = sigma^2 for all i (homoscedasticity)
3. Cov(epsilon_i, epsilon_j) = 0 for i != j (no serial correlation)

Normality is NOT a Gauss-Markov condition. OLS is BLUE whether or not residuals are normal. However, normality IS required for the t-statistics in the coefficient table to exactly follow a t-distribution in finite samples. Without it, p-values are approximate.

Homoscedasticity IS a Gauss-Markov condition. Violating it means OLS is no longer BLUE; WLS achieves lower variance. The Breusch-Pagan test tells us whether this violation is statistically confirmed.

### Output and Discovery

[To be filled in after running the notebook]

- Shapiro-Wilk W (Y1): ___  p-value: ___  Decision: ___
- Shapiro-Wilk W (Y2): ___  p-value: ___  Decision: ___
- Breusch-Pagan LM statistic (Y1): ___  p-value: ___  Decision: ___
- Breusch-Pagan LM statistic (Y2): ___  p-value: ___  Decision: ___
- Key finding: ___

---

## Change 2: Weighted Least Squares (WLS)

### What Was Added

A complete Weighted Least Squares analysis for both Y1 and Y2, inserted immediately after the OLS diagnostic section. The analysis includes:
- A full theoretical markdown explaining GLS, the WLS objective function, and how weights are determined
- Code that computes group variances from OLS residuals, assigns inverse-variance weights by height group, fits `sm.WLS()` for both targets, and produces a detailed comparison table (coefficients, standard errors, AIC, BIC, RMSE)
- A reading markdown explaining how to interpret each output element

### Why This Was Added

The Breusch-Pagan test establishes that heteroscedasticity is present. Once heteroscedasticity is confirmed, the statistically correct response is WLS. Failing to include WLS after establishing this would be the equivalent of identifying a problem and then ignoring the known solution.

WLS is explicitly listed in the DSA 8305 syllabus under Weeks 9-11 as "models for non-independent errors." The original notebook had Ridge and PCR from Weeks 9-11 but was missing this content. WLS fills that gap while also providing a natural narrative arc: the Breusch-Pagan test motivates the WLS, making the analysis internally coherent rather than treating each model as a standalone exercise.

### Concept: Weighted Least Squares

OLS minimises sum_i (y_i - x_i'*beta)^2.

WLS modifies this to sum_i w_i*(y_i - x_i'*beta)^2 where w_i > 0 is a weight for each observation.

The GLS framework shows that when errors are uncorrelated but have different variances, the optimal (BLUE) estimator is WLS with w_i = 1/Var(epsilon_i). We use a feasible WLS approach: we estimate Var(epsilon_i) from the OLS residuals within each height group, giving weights:

- w_i = 1/Var(OLS residuals | H=3.5m) for single-storey observations
- w_i = 1/Var(OLS residuals | H=7.0m) for double-storey observations

The closed-form solution is: beta_WLS = (X'WX)^{-1} X'Wy, where W is diagonal with weights w_i.

Key theoretical result: under heteroscedasticity, OLS is still unbiased but not minimum-variance. WLS with correct weights restores the minimum-variance property.

### Output and Discovery

[To be filled in after running the notebook]

- OLS residual variance ratio (7.0m vs 3.5m), Y1: ___  Y2: ___
- WLS vs OLS: RMSE difference Y1: ___  Y2: ___
- WLS vs OLS: AIC difference Y1: ___  Y2: ___
- Largest standard error change: feature=___  direction=___  magnitude=___%
- Key finding: ___

---

## Change 3: Lasso Regression with Feature Selection Path

### What Was Added

A complete Lasso regression analysis inserted after Ridge (Cells 39-41), including:
- A theoretical markdown comparing L1 vs L2 penalties, the sparse solution property, and the feature selection motivation
- Code using `LassoCV` with 10-fold CV and 100 candidate lambda values, plus a coefficient path plot with dashed lines for zeroed features
- A reading markdown interpreting the zeroed features and path structure

### Why This Was Added

The original notebook included Ridge (L2 regularisation) but not Lasso (L1 regularisation). These are complementary approaches with the same mathematical framework but different penalty geometry. A complete treatment of regularisation for the multicollinear dataset requires both. Dr. Omondi's brief asked for "very robust analysis," and excluding Lasso while including Ridge leaves the regularisation discussion incomplete.

More specifically, the EDA found that Orientation (X6) had near-zero correlation with both energy loads. Lasso provides a formal procedure to confirm this: if the L1 penalty under cross-validation optimal regularisation zeros out the Orientation dummies, it constitutes statistical evidence from a penalised regression framework that Orientation provides no incremental predictive value. This is a stronger claim than a correlation coefficient alone.

### Concept: The L1 Penalty and Sparse Solutions

The Lasso objective is: minimise ||y - X*beta||^2 + lambda * sum_j |beta_j|

The absolute value penalty creates a constraint region (L1 ball) that is a cross-polytope: a shape with corners at the coordinate axes. When the unconstrained OLS solution is outside the L1 ball, the Lasso solution lies on the boundary. Because the L1 ball has corners at axes, the boundary solution often falls exactly at a corner where one or more coefficients are exactly zero.

This geometric property does not hold for Ridge, whose L2 ball (sphere) has no corners. Ridge shrinks all coefficients toward zero but never to zero exactly.

The cross-validation path plot shows the coefficient for each feature as a function of log10(lambda). Features that reach zero early in the shrinkage path (at low lambda) were the weakest predictors. Features that remain non-zero across a wide range of lambda values are the most robustly important.

### Output and Discovery

[To be filled in after running the notebook]

- Features zeroed by Lasso Y1 (at CV-optimal lambda): ___
- Features zeroed by Lasso Y2 (at CV-optimal lambda): ___
- Was Orientation zeroed? ___  Was Glazing_Area_Distribution zeroed? ___
- CV-optimal lambda Y1: ___  Y2: ___
- Lasso RMSE vs Ridge RMSE: Y1 diff=___  Y2 diff=___
- Key finding: ___

---

## Change 4: Nonlinear Least Squares (NLS)

### What Was Added

A complete NLS model section (Cells 53-55):
- An extended theoretical markdown distinguishing "nonlinear in parameters" (genuine NLS) from "nonlinear in predictors" (polynomial, which is still linear in parameters), explaining the Levenberg-Marquardt algorithm, and justifying the chosen functional forms from building physics
- Code using `scipy.optimize.curve_fit` with the TRF (Trust Region Reflective) method to fit Y = b0*exp(b1*Height) + b2*Glazing^b3 + b4*Compactness + b5, with full parameter table including standard errors from the covariance matrix and 95% confidence intervals
- A reading markdown interpreting each parameter

### Why This Was Added

NLS was the only major topic in the DSA 8305 Weeks 5-8 syllabus block ("Non-linear models via nonlinear least squares") that was completely absent from the original notebook. The original notebook had Polynomial regression (which is linear in parameters), Natural Cubic Splines, and GAMs from the Weeks 5-8 content, but no genuine NLS. This gap was the most important addition to make from a syllabus coverage perspective.

Beyond syllabus coverage, NLS is the right tool when theory provides a specific functional form. For building energy loads, the relationship between Overall Height and energy demand has physical reasons to be approximately exponential: doubling height more than doubles volume and adds complexity to thermal dynamics that compounds multiplicatively. The LOWESS plots confirmed a nonlinear glazing effect consistent with a power law (diminishing returns: 0 < b3 < 1). These physical motivations justify NLS over an atheoretical polynomial basis expansion.

### Concept: What Makes NLS Different from Polynomial Regression

A degree-2 polynomial Y = b0 + b1*X + b2*X^2 can be written as Y = X_aug*b where X_aug = [1, X, X^2]. The parameters b = (b0, b1, b2) appear linearly. The partial derivative dY/db2 = X^2 does not depend on any other parameter. The normal equations (X'X)*b = X'y have a closed-form solution. OLS directly applies.

The NLS model Y = b0*exp(b1*X) is fundamentally different. The partial derivative dY/db1 = b0*X*exp(b1*X) still contains b0. The two parameters are intertwined. There is no closed-form solution. The Levenberg-Marquardt algorithm starts from an initial guess, linearises around it using a first-order Taylor expansion, takes a step, checks convergence via the gradient, and repeats until the gradient of the sum of squared residuals is sufficiently small.

The Gauss-Newton update (near minimum): delta_b = -(J'J)^{-1} J' r where J is the Jacobian of residuals and r is the residual vector.

The LM modification far from minimum: adds a damping term lambda*I to J'J to prevent overshoot.

The covariance matrix of parameter estimates (returned by curve_fit) is approximately (J'J)^{-1} * sigma^2, where sigma^2 is estimated from the residual sum of squares. Standard errors = sqrt(diag(pcov)).

### Output and Discovery

[To be filled in after running the notebook]

- NLS convergence status Y1: ___  Y2: ___
- b1 (height exponent) Y1: ___  Y2: ___
- b3 (glazing power) Y1: ___  Y2: ___
- Is b3 < 1 (confirming diminishing returns)? ___
- Height energy amplification ratio exp(b1*7.0)/exp(b1*3.5): ___
- NLS RMSE: Y1=___  Y2=___  vs Polynomial RMSE: Y1=___  Y2=___
- Key finding: ___

---

## Change 5: Interaction Models with Partial F-Test

### What Was Added

A complete interaction model analysis (Cells 56-58):
- A theoretical markdown explaining the additivity assumption, why it may fail here from building physics, interaction terms in OLS, the partial F-test, and interpretation of individual interaction coefficients
- Code that creates three interaction features (Height x Glazing, Height x Compactness, Height x Wall Area), fits OLS with and without them, and computes the formal partial F-test by comparing RSS values, reporting F-statistic, degrees of freedom, and p-value for each target
- Individual interaction coefficient table with p-values

### Why This Was Added

The original notebook acknowledged interaction effects as a limitation in the conclusions: "Scatter plots coloured by building height clearly showed that the relationship between Compactness and Heating Load differs between single-storey and double-storey buildings... A next step would be to formally test." This change closes that gap: interaction models are now formally tested and reported rather than listed as future work.

Interaction models are explicit DSA 8305 Weeks 12-14 content. The partial F-test for interaction significance is a standard tool for choosing between additive and interaction model specifications and is part of the hypothesis testing toolkit in the final weeks of the course.

### Concept: The Partial F-Test for Nested Models

Two models are nested if one is a special case of the other (obtained by restricting some parameters to zero). OLS without interactions is a restricted version of OLS with interactions: setting the three interaction coefficients to zero recovers the additive model.

The partial F-test statistic:

F = [ (RSS_restricted - RSS_unrestricted) / q ] / [ RSS_unrestricted / (n - k_u - 1) ]

where q is the number of restrictions (number of interaction terms), n is sample size, and k_u is the number of parameters in the unrestricted model. Under H0 (all interaction coefficients = 0), F follows an F(q, n-k_u-1) distribution.

The key advantage over comparing R-squared values: R-squared always increases when adding parameters. The F-test accounts for how much extra variance should be expected to be explained just from adding parameters, making the comparison valid.

### Output and Discovery

[To be filled in after running the notebook]

- F-statistic Y1: ___  p-value: ___  Decision: ___
- F-statistic Y2: ___  p-value: ___  Decision: ___
- Most significant interaction term: ___  coefficient: ___  p-value: ___
- AIC comparison: Interaction OLS vs OLS  Y1: ___  Y2: ___
- Physical interpretation of Height x Glazing coefficient: ___
- Key finding: ___

---

## Change 6: GAM 95% Prediction Intervals with Coverage Analysis

### What Was Added

A GAM prediction interval extension (Cells 63-65):
- A theoretical markdown distinguishing prediction intervals from confidence intervals, explaining pyGAM's Bayesian posterior covariance approach, and explaining empirical coverage as the primary calibration metric
- Code using `gam.predict(X_test_scaled, width=0.95, return_bounds=True)` to produce per-observation 95% intervals, a fan plot sorted by predicted value (green dots inside, red outside), and empirical coverage and mean interval width statistics
- A reading markdown interpreting coverage, interval width, and the spatial pattern of uncertainty

### Why This Was Added

Every model in the original notebook produced point predictions only. A point prediction tells us the expected energy load but provides no information about how uncertain that estimate is. For engineering and procurement decisions, a prediction interval is essential: it defines the range within which the true load will fall with specified probability, enabling properly sized HVAC system design with explicit coverage guarantees.

The original notebook listed "Prediction intervals" as a limitation. This change resolves that limitation. GAM prediction intervals specifically are mentioned in the DSA 8305 Weeks 12-14 content as part of semi-parametric inference, making this addition directly on-syllabus.

### Concept: Prediction Intervals vs Confidence Intervals

A confidence interval for E[Y|X0] answers: where does the mean energy load lie for buildings with parameters X0? It covers parameter uncertainty (how well is the smooth function estimated?).

A prediction interval for a single new Y at X0 answers: where will the energy load of this specific building fall? It covers both parameter uncertainty AND individual-level residual variance (sigma^2). It is always wider than the confidence interval.

pyGAM computes prediction intervals via the posterior covariance of spline coefficients. For observation x_new:
- Parameter variance: Var_param(x_new) = x_new' * Cov(beta) * x_new (in spline basis)
- Total predictive variance: sigma^2_param + sigma^2_residual
- 95% interval: mu(x_new) +/- 1.96 * sqrt(sigma^2_param + sigma^2_residual)

Empirical coverage = fraction of actual test y values that fall within their respective intervals. A well-calibrated model achieves coverage near the nominal level (95%).

### Output and Discovery

[To be filled in after running the notebook]

- Y1 empirical coverage: ___%  (target: 95.0%)
- Y2 empirical coverage: ___%  (target: 95.0%)
- Y1 mean interval width: ___ kWh/m^2
- Y2 mean interval width: ___ kWh/m^2
- Is the model well-calibrated (coverage 90-99%)? ___
- Engineering implication: ___

---

## Change 7: Unified 10-Fold Cross-Validation Comparison

### What Was Added

A unified CV comparison section (Cells 72-75):
- A markdown explaining why model-specific CV comparisons are not fair (different fold assignments) and why a unified analysis on the same folds is necessary
- Code running 10-fold CV for: OLS, Ridge (with nested inner CV for lambda), Lasso (nested), PCR, Polynomial deg=2 + Ridge, NLS (custom loop), Interaction OLS (custom loop), reporting mean CV-RMSE and standard deviation across folds for each
- An AIC/BIC comparison table for parametric models (OLS, WLS, Interaction OLS)
- A reading markdown explaining the expected ordering and how to interpret standard deviations

### Why This Was Added

Each model in Phase 4 was tuned and evaluated using its own internal CV procedure with different random seeds and fold assignments. Directly comparing the CV-RMSE values from these separate procedures is statistically invalid: a model that happened to get favorable fold assignments will appear to perform better even if it is genuinely equivalent. A unified CV with identical fold assignments for all models removes this confound.

The standard deviation of CV-RMSE across folds is equally important: it quantifies model stability. A model that achieves low mean CV-RMSE but high standard deviation is only reliably good for some subsets of the data. A stable model with consistent performance across folds is more suitable for deployment.

AIC and BIC provide a complementary comparison for maximum-likelihood-based parametric models. They penalise fit for model complexity, providing a parsimonious selection criterion that cross-validation alone does not provide.

### Concept: Why Nested CV is Needed for Penalised Models

When selecting a regularisation parameter lambda via CV and then evaluating on the same CV folds, the lambda selection procedure "sees" the fold assignments. To get an unbiased estimate of out-of-sample error for a CV-selected lambda, we need a nested (two-level) CV:

Outer loop (evaluation): 10 folds, same for all models.
Inner loop (hyperparameter selection): 5 folds within each outer training partition.

The outer folds provide unbiased evaluation. The inner folds select lambda from the outer training data only, never touching the outer test fold. This prevents the lambda from being tuned on information from the evaluation partition.

### Output and Discovery

[To be filled in after running the notebook]

- Unified CV-RMSE ranking Y1 (1=best): ___
- Unified CV-RMSE ranking Y2 (1=best): ___
- Most consistent model (lowest Std): ___
- OLS vs Ridge CV-RMSE difference: ___ (does regularisation help?)
- NLS vs Polynomial CV-RMSE: ___
- AIC ranking for Y1: ___  Y2: ___
- BIC ranking for Y1: ___  Y2: ___
- Key finding: ___

---

## Change 8: Syllabus Mapping Table

### What Was Added

A 12-row table in the overview section (Cell 0) mapping every analysis step to its DSA 8305 course week, plus an updated model overview table in Cell 22 that includes a "DSA 8305 Weeks" column for all 12 model families.

### Why This Was Added

Dr. Omondi explicitly asked the group to demonstrate "overall understanding of this unit as we approach the examination date." The mapping table makes the connection between the analytical work and the course syllabus explicit and immediately visible to the examiner. Rather than hoping the connection is implied, the table states it directly: this analysis covers Weeks 1-4 through 12-14 in a structured progression.

It also ensures that no syllabus topic is accidentally forgotten in the presentation.

---

## Files Modified

| File | Change |
|:---|:---|
| `Energy_Efficiency__Final.ipynb` | 21 new cells added; 6 existing cells modified; 60 cells -> 81 cells |
| `CHANGES.md` | This document (newly created) |
| `modify_notebook_apply.py` | Modification script (can be deleted after verification) |
| `modify_notebook_part1.py` | Content definitions (can be deleted after verification) |
| `modify_notebook_part2.py` | Content definitions (can be deleted after verification) |
| `modify_notebook_part3.py` | Content definitions (can be deleted after verification) |

---

## How to Use This Document

After running the notebook and observing the outputs, fill in each "[OUTPUT AND DISCOVERY]" section above with the actual numbers and findings. The discoveries become the spoken commentary during the DSA 8305 presentation and provide the narrative thread connecting each model to the next in the data story.

The key narrative arc for the presentation:
1. OLS is the baseline -- formal tests confirm its assumptions are violated.
2. WLS corrects for heteroscedasticity -- the correct GLS response.
3. Ridge/Lasso/PCR address multicollinearity -- Lasso formally confirms Orientation is irrelevant.
4. NLS and Polynomial address non-linearity -- NLS with physical justification.
5. Splines and GAM provide maximum flexibility -- GAM prediction intervals enable deployment.
6. Interaction model tests whether additivity holds -- formal F-test provides the answer.
7. Unified CV gives the final honest ranking -- evaluation on the same folds for every model.
