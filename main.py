import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load your datasets
combria_data = pd.read_csv('combria.csv')



# Split features and labels based on the actual target column name
X = combria_data.drop(columns=['Classification'])  # Replace 'diagnosis' with your actual target column name
y = combria_data['Classification']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nData split into training and test sets.")

# Logistic Regression Model
logreg = LogisticRegression(max_iter=10000)

# Hyperparameter Tuning using GridSearchCV
param_grid_logreg = {'C': [0.1, 1, 10, 100]}
grid_search_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5)

# Train the Logistic Regression Model
grid_search_logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = grid_search_logreg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Random Forest Model
rf = RandomForestClassifier()

# Hyperparameter Tuning using GridSearchCV
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)

# Train the Random Forest Model
grid_search_rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = grid_search_rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# Support Vector Machine Model
svm = SVC()

# Hyperparameter Tuning using GridSearchCV
param_grid_svm = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5)

# Train the SVM Model
grid_search_svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = grid_search_svm.predict(X_test)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
