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


# **Dataset informations**

The dataset contains a diverse range of health-related attributes, meticulously collected to aid in the development of predictive models for identifying individuals at risk of diabetes. There are 2768 patients with 10 columns each:

-    Id: Unique identifier for each data entry.
-    Pregnancies: Number of times pregnant.
-    Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
-    BloodPressure: Diastolic blood pressure (mm Hg).
-    SkinThickness: Triceps skinfold thickness (mm).
-    Insulin: 2-Hour serum insulin (mu U/ml).
-    BMI: Body mass index (weight in kg / height in m^2).
-    DiabetesPedigreeFunction: Diabetes pedigree function, a genetic score of diabetes.
-    Age: Age in years.
-    Outcome: Binary classification indicating the presence (1) or absence (0) of diabetes.

All credit for the dataset goes to Nandita Pore. You can find the original dataset and additional information on its official Kaggle page:
https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes

# **Aim of the Project**

The primary objective of this project is to develop and evaluate a Machine Learning classifier model to predict the presence of diabetes in subjects. We will be using the Support Vector Machine (SVM) algorithm for this classification task.

The project is purely for educational and personal purposes, providing a hands-on experience with data preprocessing, model training, and evaluation. It is not intended for use in real-world clinical or diagnostic scenarios and should never replace the expertise of qualified clinicians and medical professionals.


# **Methods**
This project was developed in Python, a widely adopted language in both academic research and industry, due to its robust ecosystem for machine learning and data science.

A Support Vector Machine (SVM) classifier was developed to address the classification problem. The model was trained and validated using a subset of the dataset. Multiple models, ranging from a simple linear SVM to more complex kernel-based models, were explored. The final model was selected based on its performance in terms of balanced accuracy and generalization capability, ensuring it avoids overfitting.

The implementation was made possible with the Scikit-learn library, an open-source Python library for machine learning. All other dependencies are listed in the project's repository.

**GitHub Repository:** [https://github.com/WPozz/diabetes-prediction-project](https://github.com/WPozz/diabetes-prediction-project)

# **Results**

The final model achieved a sensitivity of 75% and a specificity of 78% at its optimal threshold. While these metrics demonstrate an acceptable performance for a proof-of-concept, the model should be considered a supplementary tool and not a standalone diagnostic instrument.

Furthermore, a preliminary cost-benefit analysis indicates that the model could potentially lead to an estimated $779,500 in savings when integrated into a larger diagnostic workflow
