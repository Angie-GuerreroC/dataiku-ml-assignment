import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,f1_score,confusion_matrix,precision_recall_curve,auc
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from scipy.stats import randint,uniform
warnings.filterwarnings('ignore')

# Path to save visuals
VISUALS_PATH = '/Users/angieguerrero/Desktop/Dataiku Project/visuals'

# ----------------------------------------
# Step 1: Load and Clean Data
# ----------------------------------------
def load_and_clean_data(train_path, test_path):
    column_names = [
        'age', 'class_of_worker', 'detailed_industry_recode', 'detailed_occupation_recode', 'education',
        'wage_per_hour', 'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code',
        'major_occupation_code', 'race', 'hispanic_origin', 'sex', 'member_of_a_labor_union', 
        'reason_for_unemployment', 'full_or_part_time_employment_stat', 'capital_gains', 'capital_losses', 
        'dividends_from_stocks', 'tax_filer_stat', 'region_of_previous_residence', 'state_of_previous_residence',
        'detailed_household_and_family_stat', 'detailed_household_summary_in_household', 'instance_weight',
        'migration_code_change_in_msa', 'migration_code_change_in_reg', 'migration_code_move_within_reg',
        'live_in_this_house_1_year_ago', 'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
        'family_members_under_18', 'country_of_birth_father', 'country_of_birth_mother', 
        'country_of_birth_self', 'citizenship', 'own_business_or_self_employed', 
        'fill_inc_questionnaire_for_veterans_admin', 'veterans_benefits', 'weeks_worked_in_year', 
        'year', 'income'
    ]

    # Load the training and testing datasets
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Assign column names
    train_df.columns = column_names
    test_df.columns = column_names
    print("Sample of training dataset after loading:")
    print(train_df.head())
          
    # Print initial row counts
    print(f"Initial rows in training set: {train_df.shape[0]}, columns: {train_df.shape[1]}")
    print(f"Initial rows in test set: {test_df.shape[0]}, columns: {test_df.shape[1]}")

    # Drop 'instance_weight' column
    train_df.drop(columns=['instance_weight'], inplace=True)
    test_df.drop(columns=['instance_weight'], inplace=True)
    print("'instance_weight' column dropped")

    # Filter rows where age is below 18
    train_df = train_df[train_df['age'] >= 18]
    test_df = test_df[test_df['age'] >= 18]
    print(f"Rows after filtering age < 18 (train): {train_df.shape[0]}, (test): {test_df.shape[0]}")

    # Replace all identified invalid placeholders with NaN
    invalid_placeholders = ['?', '', ' ', 'NA', 'NULL']
    train_df.replace(invalid_placeholders, np.nan, inplace=True)
    test_df.replace(invalid_placeholders, np.nan, inplace=True)

    # Drop rows with missing values and duplicates
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)
    print(f"Rows after dropping missing values and duplicates (train): {train_df.shape[0]}, (test): {test_df.shape[0]}")

    # Print final row and column counts
    print(f"\nFinal rows and columns in training set: {train_df.shape}")
    print(f"Final rows and columns in testing set: {test_df.shape}")

    return train_df, test_df

# ----------------------------------------
# Additional Steps (Feature Engineering, EDA, etc.)
# ----------------------------------------
def perform_eda(train_df):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    This function generates various visualizations, including income distribution analysis and other combinations.
    """

    # Basic statistics summary
    print("Basic Statistics of Data:")
    print(train_df.describe())

    # Plot: Distribution of Income
    print("\nDistribution of Income:")
    sns.countplot(data=train_df, x='income', palette="viridis")
    plt.title('Income Distribution')
    plt.xlabel('Income (0 = Less than 50K, 1 = 50K+)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/income_distribution.png')
    plt.show()

    # Plot: Income vs Gender
    print("\nIncome vs Gender:")
    sns.countplot(data=train_df, x='sex', hue='income', palette="viridis")
    plt.title('Income by Gender')
    plt.xlabel('Gender (0 = Female, 1 = Male)')
    plt.ylabel('Count')
    plt.legend(title='Income', loc='upper right', labels=['Less than 50K', '50K+'])
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/income_by_gender.png')
    plt.show()

    # Plot: Income vs Education
    print("\nIncome vs Education:")
    sns.countplot(data=train_df, y='education', hue='income', palette="viridis", 
                  order=train_df['education'].value_counts().index)
    plt.title('Income by Education Level')
    plt.xlabel('Count')
    plt.ylabel('Education Level')
    plt.legend(title='Income', loc='upper right', labels=['Less than 50K', '50K+'])
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/income_by_education.png')
    plt.show()

    # Plot: Income vs Race
    print("\nIncome vs Race:")
    sns.countplot(data=train_df, y='race', hue='income', palette="viridis", 
                  order=train_df['race'].value_counts().index)
    plt.title('Income by Race')
    plt.xlabel('Count')
    plt.ylabel('Race')
    plt.legend(title='Income', loc='upper right', labels=['Less than 50K', '50K+'])
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/income_by_race.png')
    plt.show()

    # Plot: Age Distribution vs Income (Stacked)
    print("\nAge Distribution vs Income (Stacked):")
    sns.histplot(data=train_df, x='age', hue='income', bins=20, kde=True, palette="viridis", multiple="stack")
    plt.title('Age Distribution by Income')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(title='Income', loc='upper right', labels=['Less than 50K', '50K+'])
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/age_distribution_by_income.png')
    plt.show()

    # Plot: Age vs Income by Education Level
    print("\nAge vs Income by Education Level:")
    plt.figure(figsize=(12, 6))  # Adjusted figure size for better readability
    sns.violinplot(data=train_df, x='education', y='age', hue='income', split=True, palette="viridis")
    plt.title('Age Distribution by Education Level and Income')
    plt.xlabel('Education Level')
    plt.ylabel('Age')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better fit
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/age_by_education_income_violin.png')
    plt.show()
    
    # Plot: Age vs Income by Marital Status
    print("\nAge vs Income by Marital Status:")
    plt.figure(figsize=(12, 6))  # Adjusted figure size for better readability
    sns.violinplot(data=train_df, x='marital_stat', y='age', hue='income', split=True, palette="viridis")
    plt.title('Age Distribution by Marital Status and Income')
    plt.xlabel('Marital Status')
    plt.ylabel('Age')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better fit
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/age_by_marital_income_violin.png')
    plt.show()

    # Boxplots for Wage per Hour and Capital Gains
    print("\nBoxplots for Wage per Hour and Capital Gains:")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=train_df, x='income', y='wage_per_hour', ax=axes[0], hue='income', palette="viridis")
    axes[0].set_title('Wage per Hour by Income')
    axes[0].set_xlabel('Income (0 = Less than 50K, 1 = 50K+)')
    axes[0].set_ylabel('Wage per Hour')

    sns.boxplot(data=train_df, x='income', y='capital_gains', ax=axes[1], hue='income', palette="viridis")
    axes[1].set_title('Capital Gains by Income')
    axes[1].set_xlabel('Income (0 = Less than 50K, 1 = 50K+)')
    axes[1].set_ylabel('Capital Gains')

    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/boxplots_wage_capital.png')
    plt.show()
    
# ----------------------------------------
# Pipeline 
# ----------------------------------------
def feature_engineering(train_df, test_df):
    train_df['capital_net_gain'] = train_df['capital_gains'] - train_df['capital_losses']
    test_df['capital_net_gain'] = test_df['capital_gains'] - test_df['capital_losses']
    train_df = train_df.drop(['capital_gains', 'capital_losses'], axis=1)
    test_df = test_df.drop(['capital_gains', 'capital_losses'], axis=1)
    
    # Convert income to binary: 1 for income >= $50K, 0 for income < $50K
    train_df['income'] = train_df['income'].apply(lambda x: 1 if x.strip() == '50000+.' else 0)
    test_df['income'] = test_df['income'].apply(lambda x: 1 if x.strip() == '50000+.' else 0)

    return train_df, test_df

# def preprocess_data_for_logistic(train_df, test_df):
#     # Identify categorical columns (excluding target 'income')
#     categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
#     categorical_columns = [col for col in categorical_columns if col != 'income']

#     # One-hot encoding for categorical variables
#     train_df = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)
#     test_df = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)

#     # Align training and testing datasets
#     train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

#     # Scaling for continuous variables
#     scaler = StandardScaler()
#     continuous_columns = [col for col in train_df.select_dtypes(exclude=['object']).columns if col != 'income']
#     train_df[continuous_columns] = scaler.fit_transform(train_df[continuous_columns])
#     test_df[continuous_columns] = scaler.transform(test_df[continuous_columns])

#     return train_df, test_df

def preprocess_data_for_rf_xgb(train_df, test_df):
    # Identify categorical columns (excluding target 'income')
    categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != 'income']

    # Label encoding for categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        # Fit on training set and transform both training and testing sets
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    return train_df, test_df

# Step 4: Split Data First, Then Apply SMOTE
def split_data(train_df):
    X = train_df.drop('income', axis=1)
    y = train_df['income']

    # Train-test split (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

# Step 4: Apply SMOTE only to the training set
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

# # Step 5: Model Training (Train separate models)
# def train_logistic_regression(X_train_resampled, y_train_resampled, class_weight_dict):
#     model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict)
#     model.fit(X_train_resampled, y_train_resampled)
#     return model

def train_random_forest(X_train_resampled, y_train_resampled, class_weight_dict):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model.fit(X_train_resampled, y_train_resampled)
    return model

def train_xgboost(X_train_resampled, y_train_resampled):
    model = XGBClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    return model

# Step 6: Model Evaluation
def evaluate_model(models, X_val, y_val, model_names):
    f1_scores = []
    
    # Open a text file to save the classification reports
    report_path = f"{VISUALS_PATH}/model_classification_reports.txt"
    with open(report_path, "w") as report_file:
        report_file.write("Model Classification Reports\n")
        report_file.write("="*50 + "\n\n")

        # Iterate over each model to calculate metrics
        for model, model_name in zip(models, model_names):
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
            
            print(f"{model_name} F1-Score (Validation Set): {f1:.4f}")

            # Calculate and display precision, recall, and F1 score
            report = classification_report(y_val, y_pred)
            print(f"\n{model_name} Classification Report:\n", report)
            
            # Save the report to the file
            report_file.write(f"{model_name} Classification Report:\n")
            report_file.write(report + "\n")
            report_file.write("="*50 + "\n\n")

            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
            plt.title(f'Confusion Matrix for {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'{VISUALS_PATH}/{model_name}_confusion_matrix.png')
            plt.close()
            
            # Precision-Recall Curve
            y_probs = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, y_probs)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {model_name}')
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(f'{VISUALS_PATH}/{model_name}_precision_recall_curve.png')
            plt.close()

        if len(f1_scores) != len(model_names):
            raise ValueError("Mismatch between the number of models and the number of F1 scores")

        # Create a colormap based on the F1 scores for gradient coloring
        cmap = plt.get_cmap("viridis")  # You can choose other colormaps like 'plasma', 'inferno', etc.
        norm = plt.Normalize(min(f1_scores), max(f1_scores))  # Normalize the F1 scores to the colormap range

        # Plot Model Comparison based on F1-Score with gradient coloring
        plt.figure(figsize=(10, 6))
        bars = plt.barh(model_names, f1_scores, color=cmap(norm(f1_scores)))

        # Add a colorbar to show how the F1 scores correspond to the gradient
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='F1-Score')

        # Labeling and title
        plt.xlabel('F1-Score')
        plt.title('Model Comparison based on F1-Score')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f'{VISUALS_PATH}/model_comparison.png')
        plt.close()

# Hyperparameter tuning for XGBoost (using resampled training data)
def tune_xgboost(X_train_resampled, y_train_resampled):
    xgb_model = XGBClassifier(random_state=42)

    # Parameter grid with expanded search space
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 20),
        'subsample': uniform(0.5, 0.5),  # Uniform distribution for subsample between 0.5 and 1
        'colsample_bytree': uniform(0.5, 0.5)  # Uniform distribution for colsample_bytree
    }

    # RandomizedSearchCV with f1 scoring
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=5, 
        verbose=2, 
        scoring='f1',  # F1-score as the evaluation metric
        random_state=42, 
        n_jobs=-1  
    )

    random_search.fit(X_train_resampled, y_train_resampled)

    print("Best Parameters for XGBoost:", random_search.best_params_)
    print("Best F1-Score for XGBoost:", random_search.best_score_)

    return random_search.best_estimator_

# Evaluate the best model (XGBoost) on the test dataset (no SMOTE applied to test set)
def evaluate_best_model_on_test(best_model, X_test, y_test):
    y_pred_test = best_model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred_test)
    print(f"F1-Score on Test Set: {f1}")
    print(classification_report(y_test, y_pred_test))
    
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.title('Confusion Matrix for Best Model on Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/xgboost_confusion_matrix_comparison.png')
    plt.close()

    y_probs = best_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Best Model on Test Set')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/xgboost_precision_recall.png')
    plt.close()
    
# Plot all featues 
def plot_all_features_for_xgboost(model, X_train):
    importance = model.feature_importances_
    feature_names = X_train.columns
    print("Number of features in X_train:", len(feature_names))
    print("Number of importances from model:", len(importance))
    
    if len(importance) != len(feature_names):
        raise ValueError("Feature importance length does not match the number of features in X_train.")
    
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # Sort the features by importance
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Debugging: Print all features
    print("All features by importance:\n", feature_importance)

    # Reverse order for plotting
    feature_importance = feature_importance.iloc[::-1]

    # Plot all features
    plt.figure(figsize=(12, 8 + 0.2 * len(feature_importance)))  # Adjust height based on the number of features
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    plt.title("All Features for XGBoost")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/xgboost_all_features.png')
    plt.close()

# Plot Top 10 features for XGBoost 
def plot_top_10_features_for_xgboost(model, X_train):
    importance = model.feature_importances_
    feature_names = X_train.columns
    print("Number of features in X_train:", len(feature_names))
    print("Number of importances from model:", len(importance))
    
    if len(importance) != len(feature_names):
        raise ValueError("Feature importance length does not match the number of features in X_train.")
    
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # Sort the features by importance and take the top 10
    top_features = feature_importance.sort_values(by='Importance', ascending=False).head(10)

    # Debugging: Print top 10 features
    print("Top 10 features by importance:\n", top_features)

    # Reverse the order for plotting
    top_features = top_features.iloc[::-1]

    # Plot top 10 features
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.title("Top 10 Features for XGBoost")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{VISUALS_PATH}/xgboost_top_10_features.png')
    plt.close()

# Full Pipeline
def full_pipeline(train_path, test_path):
    # Load and clean the data
    train_df, test_df = load_and_clean_data(train_path, test_path)

    # EDA
    perform_eda(train_df)

    # Features
    train_df, test_df = feature_engineering(train_df, test_df)

    # Separate preprocessing for each model
    train_df, test_df = preprocess_data_for_rf_xgb(train_df, test_df)  # For RF and XGBoost
    # train_df, test_df = preprocess_data_for_logistic(train_df, test_df)  # For Logistic Regression

    # Split the data first (before SMOTE)
    X_train, X_val, y_train, y_val = split_data(train_df)

    # Apply SMOTE only to the training set
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Train and evaluate models (Before Tuning)
    results = {}
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # results['Logistic Regression'] = train_logistic_regression(X_train_resampled, y_train_resampled, class_weight_dict)
    results['Random Forest'] = train_random_forest(X_train_resampled, y_train_resampled, class_weight_dict)
    results['XGBoost'] = train_xgboost(X_train_resampled, y_train_resampled)

    # Plot metrics for each model (Confusion Matrix & Precision-Recall Curve)
    models = list(results.values())
    model_names = list(results.keys())
    evaluate_model(models, X_val, y_val, model_names)

    # Hyperparameter tuning for XGBoost (Best Model)
    best_xgb_model = tune_xgboost(X_train_resampled, y_train_resampled)

    # Evaluate the best model (XGBoost) on the test dataset
    X_test = test_df.drop('income', axis=1)
    y_test = test_df['income']
    evaluate_best_model_on_test(best_xgb_model, X_test, y_test)

    # Plot features for XGBoost
    plot_top_10_features_for_xgboost(best_xgb_model, X_train)
    plot_all_features_for_xgboost(best_xgb_model, X_train)

# Execute the pipeline
if __name__ == "__main__":
    train_path = '/Users/angieguerrero/Desktop/Dataiku Project/data/census_income_learn.csv'
    test_path = '/Users/angieguerrero/Desktop/Dataiku Project/data/census_income_test.csv'
    full_pipeline(train_path, test_path)
