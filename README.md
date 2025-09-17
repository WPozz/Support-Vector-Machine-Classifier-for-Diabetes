# Diabetes Prediction with Support Vector Machines (SVMs) 
This repository contains a comprehensive machine learning project for predicting diabetes using various Support Vector Machine (SVM) models. The project follows a complete end-to-end workflow, from data analysis and model training to in-depth interpretation, clinical evaluation, and deployment preparation.

The repository includes: 

- Full ML Pipeline: Implements a robust scikit-learn pipeline for data scaling and model training, ensuring consistent preprocessing.
- Hyperparameter Tuning: Utilizes GridSearchCV to find the optimal hyperparameters for Linear, Radial, and Polynomial SVMs.
- Model Evaluation: Provides detailed performance analysis using confusion matrices, classification reports, and a comparison against a majority class baseline, with a focus on     Balanced Accuracy for imbalanced datasets.

The projects values model Interpretability as it explains the model's decisions using advanced techniques:
    - Permutation Importance: Ranks features by their impact on model performance.
    - SHAP (SHapley Additive exPlanations): Explains individual predictions by showing each feature's contribution.
    - Partial Dependence Plots (PDPs): Visualizes the marginal effect of features on the predicted outcome.

There's also a dedicated clinical relevance section that evaluates the model using clinical metrics like Sensitivity, Specificity, PPV, and NPV. The section is coupled with a cost-benefit analysis and a risk stratification framework for real-world application.

Moreover, a reliability analysis is performed with Learning Curves to diagnose overfitting and Calibration Curves to ensure predicted probabilities are reliable for clinical decision-making. Lastly, the script demonstrates how to save the trained model and provides a simple FastAPI code template for a production-ready API endpoint.
