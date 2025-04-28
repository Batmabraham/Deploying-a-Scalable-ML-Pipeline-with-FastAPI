# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The Adult Income Classifier is a binary classification model (version 1.0) developed on April 28, 2025, using scikit-learn 1.5.1. It employs logistic regression with a StandardScaler for numeric features to predict whether an individual's annual income exceeds $50,000 based on census data. The model was trained with a random seed of 42 for reproducibility and a maximum of 1000 iterations to ensure convergence. The project is licensed under the MIT License.

## Intended Use
The model is intended for socioeconomic research and analysis, enabling data scientists, policy analysts, and researchers to predict income levels based on demographic and employment features. It is designed for use in exploratory data analysis or academic studies to understand income distribution patterns. The model is not suitable for real-time decision-making, automated loan approvals, or other high-stakes applications without further validation and fairness assessments.

## Training Data
The model was trained on the UCI Adult Income dataset (`census.csv`), comprising approximately 32,561 records (80% used for training, ~26,049 rows). The dataset includes 14 features: 6 numeric (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`). The target variable, `salary`, is binary (`<=50K` or `>50K`), with an imbalanced distribution (~75% `<=50K`). Missing values (denoted as `?` in `workclass`, `occupation`, and `native-country`) were imputed using the mode of each feature. Categorical features were one-hot encoded, numeric features were scaled, and the target was binarized (0 for `<=50K`, 1 for `>50K`).

## Evaluation Data
The model was evaluated on a test set consisting of 20% of the UCI Adult Income dataset (~6,512 rows). The test data underwent the same preprocessing as the training data, using the trained `OneHotEncoder`, `LabelBinarizer`, and `StandardScaler` to ensure consistency. Performance was assessed overall and on slices of categorical features to identify potential biases.

## Metrics
The model’s performance was evaluated using precision, recall, and F1-score (beta=1), with a `zero_division=1` setting to handle cases with no positive predictions. The overall test set performance is:
- **Precision**: 0.7445
- **Recall**: 0.6085
- **F1-score**: 0.6697

Performance on categorical slices was computed for each unique value of the categorical features. Example slice metrics (please update with actual metrics from `slice_output.txt`):
- **workclass: Federal-gov** (188 samples):
  - Precision: 0.8197
  - Recall: 0.7353
  - F1-score: 0.7752
- **workclass: Private** (4,595 samples):
  - Precision: 0.7381
  - Recall: 0.6245
  - F1-score: 0.6766
- **education: Bachelors** (sample placeholder):
  - Precision: 0.7500
  - Recall: 0.7000
  - F1-score: 0.7241

These metrics indicate moderate performance, with higher precision than recall, suggesting the model is conservative in predicting `>50K`. Slice performance varies, highlighting potential disparities across groups (e.g., `workclass`, `education`).


## Ethical Considerations
The model may reflect historical biases in the UCI Adult Income dataset, particularly across demographic features like `sex`, `race`, and `native-country`. For example, performance differences across `workclass` slices suggest varying reliability for different employment types, which could impact fairness in socioeconomic analyses. The imbalanced target (~75% `<=50K`) may lead to underprediction of `>50K`, disproportionately affecting certain groups. Use in high-stakes contexts (e.g., policy decisions) risks perpetuating biases without additional fairness interventions, such as demographic parity or equal opportunity metrics. Users should conduct thorough bias audits and consider fairness-aware algorithms before deployment.

## Caveats and Recommendations
The model’s performance is limited by the dataset’s imbalances and missing values, which were imputed using a simple mode-based approach. The logistic regression model assumes linear relationships and requires scaled features, which may not capture complex patterns as effectively as tree-based models (e.g., random forest). The lack of class weighting or oversampling (e.g., SMOTE) may reduce recall for the minority class (`>50K`). Recommendations include:
- Experiment with tree-based models for improved robustness.
- Implement class weighting or oversampling to address imbalance.
- Use advanced imputation techniques (e.g., KNN) for missing values.
- Validate on diverse, modern datasets to ensure generalizability.
- Conduct fairness analyses to mitigate biases in sensitive features.