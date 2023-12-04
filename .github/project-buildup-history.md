# Project Buildup History: Housing PCA Case Study

- Repository: `housing-pca-case-study`
- Category: `data_science`
- Subtype: `pca`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2023-10-16 - Day 2: Data preparation

- Task summary: Started the Housing PCA Case Study today with the data preparation phase. The dataset has about 80 features describing house characteristics and the goal is to use PCA to understand the latent structure. Did a full missing value audit — about 15 features had more than 10 percent missing, decided to impute with median for continuous and mode for categorical. Also identified and removed the ID column and any leakage-prone features before the analysis.
- Deliverable: Data prep complete. 15 features imputed. Leakage-prone columns removed.
## 2023-10-16 - Day 2: Data preparation

- Task summary: The one-hot encoding of the many categorical features exploded the feature count from 80 to 230. Noted this as something that needs careful handling in the PCA — many of the components will reflect categorical structure rather than meaningful variance.
- Deliverable: Post-encoding feature count noted. Strategy for handling categorical PCA planned.
## 2023-10-16 - Day 2: Data preparation

- Task summary: Added a feature correlation heatmap for just the continuous features to start identifying the natural groupings before running PCA.
- Deliverable: Continuous feature correlation heatmap added. Natural clusters visible in structure.
## 2023-10-23 - Day 3: PCA analysis

- Task summary: Ran the full PCA on the housing feature matrix today. The scree plot showed a meaningful drop in eigenvalue after component 10, so focused the interpretation there. The first component clearly represented overall house size. Components 2 and 3 showed a split between quality-related features and age-related features. Wrote detailed interpretations for the top five components by looking at the feature loadings.
- Deliverable: PCA run on full feature matrix. Top 5 components interpreted. First component is overall size.
## 2023-10-23 - Day 3: PCA analysis

- Task summary: Added cumulative explained variance plot. 20 components explain about 85 percent of variance — reasonable for this dataset given the number of categorical dummies.
- Deliverable: Cumulative explained variance plot added.
## 2023-12-04 - Day 4: Final write-up

- Task summary: Wrapped up the Housing PCA Case Study today. Wrote the conclusion section connecting the PCA findings back to practical implications — which components could be useful as inputs to a downstream regression, which categories of features explain most structural variance. Also added a comparison showing whether using PCA-reduced features as input to a regression model does better or worse than the full feature set.
- Deliverable: Case study concluded. PCA-vs-full features regression comparison added.
