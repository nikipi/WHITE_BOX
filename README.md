# WHITE_BOX

# Machine Learning Explainability

1. Permutation Importance

Insignificant feature can be shuffled and won't make a big impact on the model

Permutation importance is great because it created simple numeric measures to see which features mattered to a model. This helped us make comparisons between features easily, and you can present the resulting graphs to non-technical audiences.

But it doesn't tell you how each features matter. If a feature has medium permutation importance, that could mean it has

* a large effect for a few predictions, but no effect in general, or
* a medium effect for all predictions.

2. Partial Plots and GAM

3. SHAP VALUES

To understand individual predictions

Credits to https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values.
Thanks for the great tutorial

## Counterfactual Explanations
How:

Change the feature values of an instance before making the predictions and we analyze how the prediction changes

Criterion for counterfactual explanations:
1. the minimum changes to the features so that the prediction changes 

2. Change as few feature as possible

3. the change should be achieveable e.g you cannot enlarge the size of your house, even model shows 10 m2 bigger could raise the rent

## Local interpretable model-agnostic explanations(LIME)
