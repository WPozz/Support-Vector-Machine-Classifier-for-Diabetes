
#%% 0. --- Libraries import --- 

import shap
shap.initjs()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn import svm
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import ShuffleSplit
import pickle
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from scipy.stats import wilcoxon

#%% --- 1. Data loading and cleaning ---

# Using Kaggle API:
# Download the latest version of the dataset
path = kagglehub.dataset_download("nanditapore/healthcare-diabetes")

print("Path to dataset files:", path)

# Alternatively: 
data_raw = pd.read_csv('healthcare-diabetes.csv') 

# Error handling for issues like file not found 
if data_raw is None:
    raise FileNotFoundError("Dataset not found. Please check the path.")
else: 
    print("Dataset loaded successfully.")


print("Data dimensions (rows, columns):", data_raw.shape)
print(data_raw.head())
print("\nDataset info (type of data, non zero values):")
data_raw.info()
print("\n Stats:")
print(data_raw.describe())
print("\n")


# Search for missing values
print("Missing values per column:")
print(data_raw.isnull().sum())
print("\n")

if sum(data_raw.isnull().sum()) == 0:
    print("There are no missing values")
else:
    print("There are", sum(data_raw.isnull().sum()) ,"missing values")

# Define numerical columns
numerical_cols = data_raw.select_dtypes(include=np.number).columns

#%% --- 2. Exploratory data analysis (EDA) ---


# Correlation analysis 
plt.figure(figsize=(12, 10))
correlation_matrix = data_raw[numerical_cols].corr().abs()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix with every feature', y=1.02)
plt.tight_layout()
plt.show()

# Creating a function to select the 5 most important features

def get_target_correlated_features(data, target_col='Outcome', n_features=5):
    """
    Select features most correlated with the target variable.

    Inputs: 
        data (pd.DataFrame): The input DataFrame containing features and the target variable.
        target_col (str): The name of the target column. Default is 'Outcome'.
        n_features (int): The number of top correlated features to return. Default is 5.
        
    Output: 
        top_features (list): A list of the names of the top correlated features.
    """
    if target_col in data.columns:
        target_correlations = data.corr()[target_col].abs().sort_values(ascending=False)
        # Exclude the target variable itself
        top_features = target_correlations.drop(target_col).head(n_features).index.tolist()
        return top_features
    else:
        print(f"Target column '{target_col}' not found in data")
        return []


# Display the 5 most correlated features: 
print("Most correlated featires with ""Outcome"":")
top_correlated_features = get_target_correlated_features(data_raw, target_col='Outcome', n_features=5)
print(f"Selected features: {top_correlated_features}")


# Create the pair plot for the most correlated featueres with Outcome 
if len(top_correlated_features) > 1 and 'Outcome' in data_raw.columns:
    # Include the target variable for coloring
    pairplot_features = top_correlated_features + ['Outcome'] if 'Outcome' not in top_correlated_features else top_correlated_features
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(data_raw[pairplot_features], hue='Outcome', corner=True, diag_kind='hist')
    plt.suptitle('Pair Plot of Top Features Most Correlated with Target', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print correlation values with target
    print(f"\nCorrelation with target variable:")
    target_corrs = data_raw[top_correlated_features + ['Outcome']].corr()['Outcome'].drop('Outcome')
    for feature, corr in target_corrs.sort_values(ascending=False).items():
        print(f"{feature}: {corr:.3f}")
        
else:
    print("Not enough features or target variable not found for pair plot.")


# Correlation matrix with the top 5 correlated features only
plt.figure(figsize=(8, 6))
selected_corr_matrix = data_raw[top_correlated_features].corr()
sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix - Selected Top Features')
plt.tight_layout()
plt.show()

#%% --- 3. Feature Engineering: ---

# Removing the Id feature: 
data_raw = data_raw.drop('Id', axis=1)


# Create a single plot with the histograms of all the features minus Id and Outcome:
features_to_plot = [col for col in data_raw.columns if col != 'Outcome']


data_raw[features_to_plot].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle('Distribution of Features', fontsize=16)
plt.subplots_adjust(left=0.054, bottom=0.05, right=0.967, top=0.894, wspace=0.178, hspace=0.36)
plt.show()


# Apply a LabelEncoder to the 'Outcome' column.
# In this case we have only numerical data so we don't need One-hot-encoding
le = LabelEncoder()
data_raw['Outcome'] = le.fit_transform(data_raw['Outcome'])


# Data partitioning (features vs target): 

# Define the features (X) by dropping the target and other non-feature columns
# The 'Outcome' column is now encoded as 0 and 1
X = data_raw.drop('Outcome', axis=1)
y = data_raw['Outcome']


# Train-test split and pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15, stratify=y)

print(f"Dependent variable (y): '{y.name}'")
print(f"Independent variables (X): {X.columns.tolist()}")
print(f"Shape of X after encoding: {X.shape}")
print("\n")

# Data scaling and pipeline: 

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

# The pipeline can now be used as a single estimator.
# When we call `pipeline.fit()`, it will first call `fit_transform` on the scaler
# and then use the scaled data to fit the SVM classifier.

print(f"Dimensions X_train: {X_train.shape}")
print(f"Dimensions X_test: {X_test.shape}")
print(f"Dimensions y_train: {y_train.shape}")
print(f"Dimensions y_test: {y_test.shape}")
print("\n")

# Majority classifier: 
majority_class = y_train.value_counts().idxmax()
majority_accuracy = (y_test == majority_class).mean()


#%% --- 4. Model Building and Tuning ---

# Dictionary to store best models and their predictions
best_models = {}
predictions = {}
accuracies = {}


# --- Linear SVM Tuning ---
print("\n--- Tuning Linear SVM ---")
param_grid_linear = {
    'svm__C': np.logspace(-3,3,7),
    'svm__kernel': ['linear']
}

# Note: When using a Pipeline, parameters for the final estimator must be prefixed with its name in the pipeline (e.g., 'svm__C').

grid_search_linear = GridSearchCV(pipeline,
                                  param_grid_linear,
                                  cv=5,
                                  scoring='accuracy',
                                  n_jobs=-1)
grid_search_linear.fit(X_train, y_train)

best_linear_svm = grid_search_linear.best_estimator_
predictions['Linear'] = best_linear_svm.predict(X_test)
accuracies['Linear SVM'] = best_linear_svm.score(X_test, y_test)

print("Best parameters for Linear SVM:", grid_search_linear.best_params_)
print(f"Best cross-validation accuracy for Linear SVM: {grid_search_linear.best_score_:.4f}")
print(f"Test accuracy for Tuned Linear SVM: {accuracies['Linear SVM']:.4f}")
print("\nClassification Report for Tuned Linear SVM:")
print(classification_report(y_test, predictions['Linear'], zero_division=0))

# Confusion matrix: 
cm_linear = confusion_matrix(y_test, predictions['Linear'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Diabetes')
plt.ylabel('True Diabetes')
plt.title('Confusion Matrix: Tuned Linear SVM')
plt.show()


# --- Radial SVM Tuning ---
print("\n--- Tuning radial SVM ---")
param_grid_radial = {
    'svm__C': np.logspace(-3,3,7), # Regularization parameter
    'svm__gamma': ['scale', 'auto', 0.1, 1], # 'scale' uses 1 / (n_features * X.var())
    'svm__kernel': ['rbf']
}

grid_search_radial = GridSearchCV(pipeline,
                                  param_grid_radial,
                                  cv=5,
                                  scoring='accuracy',
                                  n_jobs=-1)
grid_search_radial.fit(X_train, y_train)

best_radial_svm = grid_search_radial.best_estimator_
predictions['Radial'] = best_radial_svm.predict(X_test)
accuracies['Radial SVM'] = best_radial_svm.score(X_test, y_test)

print("Best parameters for Radial SVM:", grid_search_radial.best_params_)
print(f"Best cross-validation accuracy for Radial SVM: {grid_search_radial.best_score_:.4f}")
print(f"Test accuracy for Tuned Radial SVM: {accuracies['Radial SVM']:.4f}")
print("\nClassification Report for Tuned Radial SVM:")
print(classification_report(y_test, predictions['Radial'], zero_division=0))


# Confusion matrix: 
cm_radial = confusion_matrix(y_test, predictions['Radial'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm_radial, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Diabetes')
plt.ylabel('True Diabetes')
plt.title('Confusion Matrix: Tuned Radial SVM')
plt.show()


# Polynomial SVM Tuning 
print("\n--- Tuning Polynomial SVM ---")
param_grid_poly = {
    'svm__C': np.logspace(-3,3,7), # Regularization parameter
    'svm__degree': [2, 3, 4], # Polynomial degree
    'svm__gamma': ['scale', 'auto'], # 'scale' uses 1 / n_features * X.var())
    'svm__kernel': ['poly']
}

grid_search_poly = GridSearchCV(pipeline,
                                param_grid_poly,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1)
grid_search_poly.fit(X_train, y_train)

best_poly_svm = grid_search_poly.best_estimator_
predictions['Polynomial'] = best_poly_svm.predict(X_test)
accuracies['Polynomial SVM'] = best_poly_svm.score(X_test, y_test)

print("Best parameters for Polynomial SVM:", grid_search_poly.best_params_)
print(f"Best cross-validation accuracy for Polynomial SVM: {grid_search_poly.best_score_:.4f}")
print(f"Test accuracy for Tuned Polynomial SVM: {accuracies['Polynomial SVM']:.4f}")
print("\nClassification Report for Tuned Polynomial SVM:")
print(classification_report(y_test, predictions['Polynomial'], zero_division=0))

# Confusion matrix
cm_poly = confusion_matrix(y_test, predictions['Polynomial'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Diabetes')
plt.ylabel('True Diabetes')
plt.title('Confusion Matrix: Tuned Polynomial SVM')
plt.show()

#%% --- 5. Model Validation --- 

# Store best models 
best_models['Linear'] = best_linear_svm
best_models['Radial'] = best_radial_svm
best_models['Polynomial'] = best_poly_svm

def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 10), random_state=45):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding (train, test) splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` is used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    train_sizes : array-like, shape (n_ticks,), dtype in (float, int)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum number of samples (which is the number of samples
        in the training set). If the dtype is int, it is regarded as absolute
        numbers.

    random_state : int, RandomState instance or None, optional (default=45)
        Controls the randomness of the cross-validation splits.
        If int, it is used as a seed for `ShuffleSplit`.
        If `RandomState`, it is used directly. If `None`, `RandomState` is
        used without a seed.
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, random_state=45, return_times=False)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(f'Learning Curves: {title}')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Add diagnostic text
    final_gap = train_scores_mean[-1] - test_scores_mean[-1]
    if final_gap > 0.05:
        plt.text(0.02, 0.02, f'Potential Overfitting\nGap: {final_gap:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        plt.text(0.02, 0.02, f'Good Generalization\nGap: {final_gap:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.show()



print("\n--- Learning Curves Analysis ---")
# Plot learning curves for best models
for model_name, model in [('Radial SVM', best_radial_svm), 
                         ('Linear SVM', best_linear_svm),
                         ('Polynomial SVM', best_poly_svm)]:
    plot_learning_curve(model, model_name, X_train, y_train)


# Select the best model based on Balanced Accuracy:
# Balanced accuracy = (Sensitivity + Specificity) / 2
# Let's build a dataframe with each specificity and sensitivity for each model: 

sensitivity = []
specificity = []
for model_name, model in best_models.items():
    cm = confusion_matrix(y_test, model.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    sensitivity.append(tp / (tp + fn))
    specificity.append(tn / (tn + fp))

# Calculate Balanced accuracy for each model: 

balanced_accuracy = [(sensitivity[i] + specificity[i]) / 2 for i in range(len(best_models))]
balanced_accuracy_df = pd.DataFrame({
    'Model': list(best_models.keys()),
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'Balanced Accuracy': balanced_accuracy
})


# Model selection: 
best_model = balanced_accuracy_df.loc[balanced_accuracy_df['Balanced Accuracy'].idxmax(), 'Model'] 
print(f"According to the balanced accuracy, the best model is: {best_model}")

# Plot balanced accuracy with a barplot for each model:
plt.figure(figsize=(10, 7))
bars = plt.bar(balanced_accuracy_df['Model'], balanced_accuracy_df['Balanced Accuracy'], color=['blue', 'green', 'orange'])
plt.xlabel('SVM Model Type')
plt.ylabel('Balanced Accuracy')
plt.title('Comparison of Tuned SVM Model Balanced Accuracies vs. Majority Classifier', y=1.05) # Adjusted y-position
plt.ylim(0.5, 1) # Set a more appropriate y-limit to better visualize differences

# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

# Add a horizontal line for the Majority Classifier baseline for comparison
plt.axhline(y=majority_accuracy, color='red', linestyle='--', label=f'Majority Class Baseline: {majority_accuracy:.4f}')
plt.legend()
plt.show()


#%% --- 7. Fine-tuning the Polynomial SVM --- 

# We reduce the cost intervall to generalize more and to avoid overfitting. 
# --- Polynomial SVM Fine-Tuning (Focused search around best parameters) ---
print(f"Initial best parameters were: {grid_search_poly.best_params_}")
print("Now fine-tuning with increased regularization to reduce overfitting...")

param_grid_poly_finetuned = {
    'svm__C': np.logspace(-2, 2, 30),  
    'svm__degree': [2, 3],             
    'svm__gamma': ['scale', 'auto'],  
    'svm__kernel': ['poly']
}

grid_search_poly_finetuned = GridSearchCV(pipeline,
                                          param_grid_poly_finetuned,
                                          cv=5,
                                          scoring='accuracy',
                                          n_jobs=-1)
grid_search_poly_finetuned.fit(X_train, y_train)

best_poly_svm_finetuned = grid_search_poly_finetuned.best_estimator_
predictions['Polynomial_Finetuned'] = best_poly_svm_finetuned.predict(X_test)
accuracies['Polynomial SVM Finetuned'] = best_poly_svm_finetuned.score(X_test, y_test)

print("Best parameters for Fine-tuned Polynomial SVM:", grid_search_poly_finetuned.best_params_)
print(f"Best cross-validation accuracy for Fine-tuned Polynomial SVM: {grid_search_poly_finetuned.best_score_:.4f}")
print(f"Test accuracy for Fine-tuned Polynomial SVM: {accuracies['Polynomial SVM Finetuned']:.4f}")
print("\nClassification Report for Fine-tuned Polynomial SVM:")
print(classification_report(y_test, predictions['Polynomial_Finetuned'], zero_division=0))

# Confusion matrix
cm_poly_finetuned = confusion_matrix(y_test, predictions['Polynomial_Finetuned'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm_poly_finetuned, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Diabetes')
plt.ylabel('True Diabetes')
plt.title('Confusion Matrix: Fine-tuned Polynomial SVM')
plt.show()

# Update best_models dictionary
best_models['Polynomial_Finetuned'] = best_poly_svm_finetuned


# Plot the learning curve for the Fine-Tuned Polynomial SVM : 
plot_learning_curve( best_poly_svm_finetuned, f'Polynomial SVM - Fine-Tuned Regularization', X_train, y_train)


#%% Three-Level Regularization Fine-Tuning

# --- Low, Medium, Hard Regularization for Polynomial SVM ---
print("\n--- Three-Level Regularization Fine-Tuning ---")

# Define regularization levels
# Let's use the best C for the fine-tuned model as starting point 
regularization_configs = {
    'Low': {
        'svm__C': np.logspace(-2, 2, 20),    # Less regularization
        'svm__degree': [2, 3],
        'svm__gamma': ['scale', 'auto'],
        'svm__kernel': ['poly']
    },
    'Medium': {
        'svm__C': np.logspace(-3, 1, 20),   # Moderate regularization  
        'svm__degree': [2, 3],
        'svm__gamma': ['scale', 'auto'],
        'svm__kernel': ['poly']
    },
    'Hard': {
        'svm__C': np.logspace(-4, 0, 20),   # Strong regularization
        'svm__degree': [2, 3],
        'svm__gamma': ['scale', 'auto'], 
        'svm__kernel': ['poly']
    }
}

# Store results for comparison
regularization_results = {}
regularization_models = {}

for reg_level, param_grid in regularization_configs.items():
    print(f"\n--- Training {reg_level} Regularization Model ---")
    print(f"C range: {param_grid['svm__C']}")
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    cv_accuracy = grid_search.best_score_
    
    # Store results
    regularization_models[f'Poly_{reg_level}'] = best_model
    regularization_results[reg_level] = {
        'best_params': grid_search.best_params_,
        'cv_accuracy': cv_accuracy,
        'test_accuracy': test_accuracy,
        'model': best_model,
        'grid_search_object': grid_search

    }
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"CV accuracy: {cv_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

# Compare all three regularization levels
print("\n" + "="*60)
print("REGULARIZATION COMPARISON")
print("="*60)
print("Level    | CV Acc  | Test Acc | Gap     | Best C")
print("-" * 50)

for level, results in regularization_results.items():
    gap = results['cv_accuracy'] - results['test_accuracy']
    best_c = results['best_params']['svm__C']
    print(f"{level:8} | {results['cv_accuracy']:.4f} | {results['test_accuracy']:.4f}  | {gap:+.4f} | {best_c:.3f}")

#%%
# Plot learning curves for all three
for level, results in regularization_results.items():
    plot_learning_curve(results['model'], f'Polynomial SVM - {level} Regularization', X_train, y_train)

# Select best regularization model based on smallest CV-Test gap (best generalization)
best_reg_level = min(regularization_results.keys(), 
                     key=lambda x: abs(regularization_results[x]['cv_accuracy'] - regularization_results[x]['test_accuracy']))
best_regularized_model = regularization_results[best_reg_level]['model']
best_regularized_gride_search = regularization_results[best_reg_level]['grid_search_object']

print(f"\n🏆 Best Regularization Level: {best_reg_level}")
print(f"Selected for smallest CV-Test gap (best generalization)")
print(f"Final model test accuracy: {regularization_results[best_reg_level]['test_accuracy']:.4f}")

# Add best regularized model to main comparison
best_models[f'Poly_Best_Regularized'] = best_regularized_model

# Update the balanced accuracy comparison to include the fine-tuned model
sensitivity_finetuned = []
specificity_finetuned = []
for model_name, model in best_models.items():
    cm = confusion_matrix(y_test, model.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    sensitivity_finetuned.append(tp / (tp + fn))
    specificity_finetuned.append(tn / (tn + fp))

balanced_accuracy_finetuned = [(sensitivity_finetuned[i] + specificity_finetuned[i]) / 2 for i in range(len(best_models))]
balanced_accuracy_df_updated = pd.DataFrame({
    'Model': list(best_models.keys()),
    'Sensitivity': sensitivity_finetuned,
    'Specificity': specificity_finetuned,
    'Balanced Accuracy': balanced_accuracy_finetuned
})

print("\nUpdated Model Comparison with Fine-tuned Polynomial:")
print(balanced_accuracy_df_updated)

# Plot updated comparison
plt.figure(figsize=(12, 7))
bars = plt.bar(balanced_accuracy_df_updated['Model'], balanced_accuracy_df_updated['Balanced Accuracy'], 
               color=['blue', 'green', 'orange', 'purple'])
plt.xlabel('SVM Model Type')
plt.ylabel('Balanced Accuracy')
plt.title('Updated Comparison: All SVM Models Including Fine-tuned Polynomial')
plt.ylim(0.5, 1)
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

plt.axhline(y=majority_accuracy, color='red', linestyle='--', label=f'Majority Class Baseline: {majority_accuracy:.4f}')
plt.legend()
plt.tight_layout()
plt.show()


#%% Threshold Tuning for Sensitivity-Specificity Optimization
print("\n" + "="*60)
print("THRESHOLD TUNING FOR SENSITIVITY-SPECIFICITY OPTIMIZATION")
print("="*60)

# Get predicted probabilities for the best model
y_prob_best = best_regularized_model.predict_proba(X_test)[:, 1]

# Define threshold range
thresholds = np.arange(0.1, 1.0, 0.05)

# Calculate sensitivity and specificity for each threshold
sensitivity_scores = []
specificity_scores = []
threshold_results = []

print("Threshold | Sensitivity | Specificity | F1-Score | Flagged Patients")
print("-" * 65)

for threshold in thresholds:
    # Apply threshold to get predictions
    y_pred_thresh = (y_prob_best >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    flagged = np.sum(y_pred_thresh)
    
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
    threshold_results.append({
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'flagged': flagged
    })
    
    print(f"{threshold:^9.2f} | {sensitivity:^11.3f} | {specificity:^11.3f} | {f1:^8.3f} | {flagged:^15}")

# Plot ROC Curve and Sensitivity-Specificity Trade-off
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title('ROC Curve')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Sensitivity-Specificity vs Threshold
ax2.plot(thresholds, sensitivity_scores, 'b-', label='Sensitivity', linewidth=2)
ax2.plot(thresholds, specificity_scores, 'r-', label='Specificity', linewidth=2)
ax2.set_xlabel('Classification Threshold')
ax2.set_ylabel('Score')
ax2.set_title('Sensitivity vs Specificity by Threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.1, 0.95])

plt.tight_layout()
plt.show()

# Find optimal thresholds for different clinical scenarios
print("\nOPTIMAL THRESHOLDS FOR DIFFERENT CLINICAL SCENARIOS:")
print("="*60)

# Scenario 1: Maximize sensitivity (screening tool)
high_sensitivity_idx = np.argmax([r['sensitivity'] for r in threshold_results])
best_sensitivity = threshold_results[high_sensitivity_idx]

# Scenario 2: Balance sensitivity and specificity (Youden's J statistic)
j_scores = [(r['sensitivity'] + r['specificity'] - 1) for r in threshold_results]
balanced_idx = np.argmax(j_scores)
balanced_threshold = threshold_results[balanced_idx]

# Scenario 3: Achieve at least 80% sensitivity
min_sensitivity_80 = [r for r in threshold_results if r['sensitivity'] >= 0.80]
if min_sensitivity_80:
    sens_80_threshold = max(min_sensitivity_80, key=lambda x: x['specificity'])
else:
    sens_80_threshold = None

print(f"1. SCREENING SCENARIO (Maximize Sensitivity):")
print(f"   Threshold: {best_sensitivity['threshold']:.2f}")
print(f"   Sensitivity: {best_sensitivity['sensitivity']:.3f} | Specificity: {best_sensitivity['specificity']:.3f}")
print(f"   Clinical impact: Catches {best_sensitivity['sensitivity']*100:.1f}% of diabetic patients")
print()

print(f"2. BALANCED SCENARIO (Youden's J Index):")
print(f"   Threshold: {balanced_threshold['threshold']:.2f}")
print(f"   Sensitivity: {balanced_threshold['sensitivity']:.3f} | Specificity: {balanced_threshold['specificity']:.3f}")
print(f"   J-statistic: {j_scores[balanced_idx]:.3f}")
print()

if sens_80_threshold:
    print(f"3. CLINICAL TARGET (≥80% Sensitivity):")
    print(f"   Threshold: {sens_80_threshold['threshold']:.2f}")
    print(f"   Sensitivity: {sens_80_threshold['sensitivity']:.3f} | Specificity: {sens_80_threshold['specificity']:.3f}")
    print(f"   Clinical impact: Catches {sens_80_threshold['sensitivity']*100:.1f}% of diabetic patients")
    print(f"                   {sens_80_threshold['flagged']} out of {len(y_test)} patients flagged for follow-up")
else:
    print("3. CLINICAL TARGET (≥80% Sensitivity): Not achievable with current model")

print()
print("THRESHOLD RECOMMENDATION:")
print("="*30)
default_threshold = 0.5
default_pred = (y_prob_best >= default_threshold).astype(int)
tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_test, default_pred).ravel()
default_sensitivity = tp_def / (tp_def + fn_def)
default_specificity = tn_def / (tn_def + fp_def)

print(f"Default threshold (0.50): Sensitivity = {default_sensitivity:.3f}, Specificity = {default_specificity:.3f}")
print(f"Recommended threshold ({balanced_threshold['threshold']:.2f}): Sensitivity = {balanced_threshold['sensitivity']:.3f}, Specificity = {balanced_threshold['specificity']:.3f}")
print(f"Improvement: +{balanced_threshold['sensitivity'] - default_sensitivity:.3f} sensitivity, {balanced_threshold['specificity'] - default_specificity:+.3f} specificity")



#%% Statistical Significance Testing
print("\n" + "="*60 + "\nSTATISTICAL SIGNIFICANCE TESTING" + "\n" + "="* 60)

# Extract CV scores from GridSearchCV objects
models_cv_scores = {
    'Linear SVM': grid_search_linear.cv_results_['split0_test_score'] + 
                  grid_search_linear.cv_results_['split1_test_score'] + 
                  grid_search_linear.cv_results_['split2_test_score'] + 
                  grid_search_linear.cv_results_['split3_test_score'] + 
                  grid_search_linear.cv_results_['split4_test_score'],
    'Radial SVM': grid_search_radial.cv_results_['split0_test_score'] + 
                  grid_search_radial.cv_results_['split1_test_score'] + 
                  grid_search_radial.cv_results_['split2_test_score'] + 
                  grid_search_radial.cv_results_['split3_test_score'] + 
                  grid_search_radial.cv_results_['split4_test_score'],
    'Polynomial SVM': grid_search_poly.cv_results_['split0_test_score'] + 
                      grid_search_poly.cv_results_['split1_test_score'] + 
                      grid_search_poly.cv_results_['split2_test_score'] + 
                      grid_search_poly.cv_results_['split3_test_score'] + 
                      grid_search_poly.cv_results_['split4_test_score'],
    'Polynomial Finetuned': grid_search_poly_finetuned.cv_results_['split0_test_score'] + 
                           grid_search_poly_finetuned.cv_results_['split1_test_score'] + 
                           grid_search_poly_finetuned.cv_results_['split2_test_score'] + 
                           grid_search_poly_finetuned.cv_results_['split3_test_score'] + 
                           grid_search_poly_finetuned.cv_results_['split4_test_score'],
    'Polynomial Best Regularized':  best_regularized_gride_search.cv_results_['split0_test_score'] + 
                           best_regularized_gride_search.cv_results_['split1_test_score'] + 
                           best_regularized_gride_search.cv_results_['split2_test_score'] + 
                           best_regularized_gride_search.cv_results_['split3_test_score'] + 
                           best_regularized_gride_search.cv_results_['split4_test_score'],
                           
}


# Get the best CV scores for each model (5 folds each)
best_cv_scores = {}
for model_name, grid_search in [
    ('Linear SVM', grid_search_linear),
    ('Radial SVM', grid_search_radial), 
    ('Polynomial SVM', grid_search_poly),
    ('Polynomial Finetuned', grid_search_poly_finetuned),
    ('Polynomial Best Regularized', best_regularized_gride_search)
]:
    best_idx = grid_search.best_index_
    scores = [
        grid_search.cv_results_[f'split{i}_test_score'][best_idx] 
        for i in range(5)
    ]
    best_cv_scores[model_name] = scores
    print(f"{model_name} CV scores: {[f'{s:.4f}' for s in scores]} (mean: {np.mean(scores):.4f})")

# Perform pairwise Wilcoxon signed-rank tests
print("\nPAIRWISE STATISTICAL SIGNIFICANCE TESTS:")
print("(Wilcoxon signed-rank test, two-tailed)")
print("-" * 60)

model_names = list(best_cv_scores.keys())
for i, model1 in enumerate(model_names):
    for j, model2 in enumerate(model_names[i+1:], i+1):
        scores1 = np.array(best_cv_scores[model1])
        scores2 = np.array(best_cv_scores[model2])
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
        
        mean_diff = np.mean(scores1) - np.mean(scores2)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{model1} vs {model2}:")
        print(f"  Mean difference: {mean_diff:+.4f}")
        print(f"  p-value: {p_value:.4f} {significance}")
        print(f"  {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'}")
        print()

print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


#%%

# --- Calculate Permutation Feature Importance ---

result = permutation_importance(best_regularized_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# --- Organize the results into a DataFrame ---
feature_importances_regularized_svm = pd.DataFrame({
    # Access the key with bracket notation
    'Importance': result['importances_mean'],
    'Feature': X.columns
}).sort_values(by='Importance', ascending=False)

# --- Plot the results ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importances_regularized_svm, palette='viridis', legend=False)
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Polynomial Regularized SVM (Permutation Importance)')
plt.show()


#%% SHAP Analysis for the Best Regularized Model

print("\n" + "="*50)
print("SHAP ANALYSIS FOR BEST REGULARIZED MODEL")
print("="*50)

# The model chosen for SHAP analysis is the one that best balances
# performance and generalization from your previous analysis.
best_model_for_shap = best_regularized_model

# Use a small, representative subset of the training data as the background
# This is crucial for KernelExplainer and speeds up computation.
background_data = shap.sample(X_train, 100)

# Extract the final estimator and the scaler from the pipeline
final_estimator = best_model_for_shap.named_steps['svm']
scaler = best_model_for_shap.named_steps['scaler']

# Create a custom prediction function that handles the preprocessing
def predict_shap(X_input):
    """
    Custom predict function for SHAP that scales the data first.
    
    Args:
        X_input (np.array): Input features as a numpy array.
    
    Returns:
        np.array: Predicted probabilities for the positive class.
    """
    # Ensure the input is a DataFrame with correct feature names for the scaler
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    X_scaled = scaler.transform(X_df)
    # The final estimator returns a numpy array, which is what SHAP expects.
    return final_estimator.predict_proba(X_scaled)[:, 1]

# Create the SHAP explainer using the custom wrapper function and background data.
explainer = shap.KernelExplainer(predict_shap, background_data)

# Calculate SHAP values for a subset of the test data.
# A small subset is enough to demonstrate the concept and is much faster.
X_test_subset = X_test.sample(n=min(100, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_test_subset)

# Plot the SHAP summary plot.
print("\nGenerating SHAP Summary Plot...")
# Use the X_test_subset DataFrame to ensure feature names are displayed.
shap.summary_plot(
    shap_values,
    X_test_subset,
    feature_names=X_test_subset.columns.tolist(),
    plot_type="bar",
    show=False # Use plt.show() later for better control
)
plt.title('SHAP Summary Plot for Best Regularized Polynomial Model')
plt.tight_layout()
plt.show()

# Generate a force plot for a single, illustrative instance.
print("Generating SHAP Force Plot for a sample instance...")

# Pick a random instance from the test set for demonstration
random_instance_idx = np.random.randint(0, len(X_test_subset))
random_instance_data = X_test_subset.iloc[[random_instance_idx]]

# Get the SHAP values for that instance
shap_values_instance = explainer.shap_values(random_instance_data)

# Get the expected value (base value) for the plot
expected_value_to_plot = explainer.expected_value

# Create and save the interactive plot in HTML format
html_output = shap.force_plot(
    expected_value_to_plot,
    shap_values_instance,
    random_instance_data,
    link="logit", # Use logit link for binary classification probability
    matplotlib=False
)

shap.save_html("force_plot.html", html_output)
print("Interactive force plot saved as 'force_plot.html'.")

# Add a more informative summary plot
print("\nGenerating SHAP Beeswarm Plot (more detailed summary)...")
shap.summary_plot(
    shap_values,
    X_test_subset,
    feature_names=X_test_subset.columns.tolist(),
    show=False # Use plt.show() later
)
plt.title('SHAP Beeswarm Plot for Best Regularized Polynomial Model')
plt.tight_layout()
plt.show()


#%%

# --- 6. Calibration Analysis ---

print("\n--- Calibration Analysis ---")

def plot_calibration_curve(models_dict, X_test, y_test):
    """Plot calibration curve for multiple models."""
    plt.figure(figsize=(12, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
    
    for model_name, model in models_dict.items():
        # Get predicted probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f'{model_name}', linewidth=2, markersize=8)
        
        # Calculate Brier Score (lower is better)
        brier_score = np.mean((y_prob - y_test) ** 2)
        print(f"Brier Score for {model_name}: {brier_score:.4f}")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot - How Well Do Predicted Probabilities Match Reality?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add interpretation text
    plt.text(0.02, 0.98, 'Closer to diagonal = Better calibrated\nLower Brier Score = Better', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# Plot calibration for all models
calibration_models = {
    'Radial SVM': best_radial_svm,
    'Linear SVM': best_linear_svm,
    'Polynomial SVM': best_poly_svm,
    'Polynomial SVM Finetuned': best_poly_svm_finetuned,
    'Best Regularized Polynomial': best_regularized_model
}
plot_calibration_curve(calibration_models, X_test, y_test)

# Create calibrated version of best model
print("\n--- Creating Calibrated Model ---")
calibrated_clf = CalibratedClassifierCV(best_regularized_model.named_steps['svm'], method='sigmoid', cv=3)

# Fit calibrated classifier on scaled training data
X_train_scaled = best_regularized_model.named_steps['scaler'].transform(X_train)
calibrated_clf.fit(X_train_scaled, y_train)

# Compare original vs calibrated
X_test_scaled = best_regularized_model.named_steps['scaler'].transform(X_test)
original_probs = best_regularized_model.predict_proba(X_test)[:, 1]
calibrated_probs = calibrated_clf.predict_proba(X_test_scaled)[:, 1]

original_brier = np.mean((original_probs - y_test) ** 2)
calibrated_brier = np.mean((calibrated_probs - y_test) ** 2)

print(f"Original Brier Score: {original_brier:.4f}")
print(f"Calibrated Brier Score: {calibrated_brier:.4f}")
print(f"Improvement: {original_brier - calibrated_brier:.4f}")

#%%

# --- 7. Partial Dependence Plots ---
print("\n--- Partial Dependence Analysis ---")

# Get top 4 most important features
top_features = feature_importances_regularized_svm.head(4)['Feature'].tolist()
print(f"Creating PDPs for top features: {top_features}")

# 1D Partial Dependence Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    PartialDependenceDisplay.from_estimator(
        best_regularized_model, X_test, [feature], 
        ax=axes[i], grid_resolution=50
    )
    axes[i].set_title(f'PDP: {feature}')

plt.suptitle('Partial Dependence Plots - Top 4 Features', fontsize=16)
plt.tight_layout()
plt.show()

# 2D Partial Dependence Plot (Feature Interactions)
print("Creating 2D PDP for top 2 feature interaction...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
PartialDependenceDisplay.from_estimator(
    best_regularized_model, X_test, [top_features[:2]], 
    ax=ax, grid_resolution=20
)
plt.title(f'2D PDP: {top_features[0]} vs {top_features[1]} Interaction')
plt.tight_layout()
plt.show()
#%%

# --- 8. Clinical Impact Analysis ---
print("\n" + "="*60)
print("CLINICAL IMPACT ANALYSIS")
print("="*60)

# Calculate clinical metrics using the balanced threshold
y_pred_best = best_regularized_model.predict(X_test)
y_prob_best = best_regularized_model.predict_proba(X_test)[:, 1]

# Apply the balanced threshold for clinical impact analysis
y_pred_best = (y_prob_best >= balanced_threshold['threshold']).astype(int)

# Confusion matrix components
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

# Clinical metrics
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
ppv = tp / (tp + fp)  # Positive Predictive Value
npv = tn / (tn + fn)  # Negative Predictive Value

# Cost-benefit analysis (estimates values)
cost_false_negative = 4000  # Cost of missing diabetes diagnosis
cost_false_positive = 700    # Cost of unnecessary follow-up
cost_screening = 50          # Cost of screening test

total_cost = (fn * cost_false_negative + 
              fp * cost_false_positive + 
              len(y_test) * cost_screening)

cost_without_model = len(y_test) * cost_screening + sum(y_test) * cost_false_negative

print(f"""
CLINICAL PERFORMANCE METRICS (With balanced threshold):
┌─────────────────────────────────┬─────────────┐
│ Metric                          │ Value       │
├─────────────────────────────────┼─────────────┤
│ Sensitivity (Recall)            │ {sensitivity:.3f}       │
│ Specificity                     │ {specificity:.3f}       │
│ Positive Predictive Value (PPV) │ {ppv:.3f}       │
│ Negative Predictive Value (NPV) │ {npv:.3f}       │
│ F1-Score                        │ {2*tp/(2*tp+fp+fn):.3f}       │
└─────────────────────────────────┴─────────────┘

CLINICAL INTERPRETATION:
• Out of 100 patients WITH diabetes, {int(sensitivity*100)} will be correctly identified
• Out of 100 patients WITHOUT diabetes, {int(specificity*100)} will be correctly classified  
• When model predicts diabetes, it's correct {int(ppv*100)}% of the time
• When model predicts no diabetes, it's correct {int(npv*100)}% of the time

COST-BENEFIT ANALYSIS (Example):
• Model-assisted screening cost: ${total_cost:,.0f}
• Cost without model: ${cost_without_model:,.0f}
• Potential savings: ${cost_without_model - total_cost:,.0f}
• False negatives (missed cases): {fn} patients
• False positives (unnecessary follow-ups): {fp} patients

CLINICAL RECOMMENDATIONS:
• Model shows {'good' if sensitivity > 0.8 else 'moderate'} sensitivity - suitable for screening
• {'High' if specificity > 0.8 else 'Moderate'} specificity reduces unnecessary referrals
• Consider ensemble with other diagnostic tools for critical decisions
• Regular model retraining recommended with new patient data
""")

# Risk stratification
print("\nRISK STRATIFICATION ANALYSIS:")
risk_thresholds = [0.3, 0.5, 0.7, 0.9]
print("Threshold | Sensitivity | Specificity | PPV   | NPV   | Patients Flagged")
print("-" * 70)

for threshold in risk_thresholds:
    y_pred_thresh = (y_prob_best >= threshold).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_thresh).ravel()
    
    sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    ppv_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    npv_t = tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
    flagged = sum(y_pred_thresh)
    
    print(f"{threshold:^9} | {sens_t:^11.3f} | {spec_t:^11.3f} | {ppv_t:^5.3f} | {npv_t:^5.3f} | {flagged:^15}")

#%%

# --- 9. Model Deployment Example ---
print("\n" + "="*60)
print("MODEL DEPLOYMENT PREPARATION")
print("="*60)

# Save the model and preprocessing pipeline
model_artifacts = {
    'model': best_regularized_model,
    'feature_names': X.columns.tolist(),
    'label_encoder': le,
    'model_metadata': {
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'final_model_test_accuracy': regularization_results[best_reg_level]['test_accuracy'],
    'best_parameters': regularization_results[best_reg_level]['best_params'],
    'feature_importance': feature_importances_regularized_svm.to_dict('records'),
    'recommended_threshold': balanced_threshold['threshold']
}
}

# Save model
with open('diabetes_svm_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("✓ Model saved as 'diabetes_svm_model.pkl'")

# Create a simple prediction function

def predict_diabetes_risk(patient_data, model, threshold): # Pass model and threshold in
    """Predict diabetes risk using the optimized threshold.""
    
    Predict diabetes risk for a single patient.
    
    Args:
        patient_data (dict): Patient features as key-value pairs

        model (sklearn.pipeline.Pipeline): The trained SVM pipeline model.

        threshold (float): The optimized probability threshold for classification.
        
    Returns:
        dict: Prediction results with probability and risk level
    """

    patient_df = pd.DataFrame([patient_data])
    
    # Get probability
    probability = model.predict_proba(patient_df)[0, 1]
    
    # Make prediction using the reccomended threshold
    prediction_class = 1 if probability >= threshold else 0
    prediction_label = 'Diabetes' if prediction_class == 1 else 'No Diabetes'

    # Stratify risk based on the probability score
    if probability >= 0.7:
        risk_level = "High Risk"
        recommendation = "Immediate medical consultation recommended"
    elif probability >= 0.5:
        risk_level = "Moderate Risk" 
        recommendation = "Schedule follow-up within 3 months"
    else: # You could align this with your recommended threshold
        risk_level = "Low to Moderate Risk"
        recommendation = "Follow up with annual screening"
    
    return {
        'diabetes_probability': round(probability, 3),
        'prediction_at_recommended_threshold': prediction_label,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

# Example Usage:
# loaded_artifacts = pickle.load(open('diabetes_svm_model.pkl', 'rb'))
# final_model = loaded_artifacts['model']
# final_threshold = loaded_artifacts['reccomended_threshold']
# predict_diabetes_risk(some_patient_data, final_model, final_threshold)

  
# %%
