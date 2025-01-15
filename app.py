from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load your datasets
combria_data = pd.read_csv('combria.csv')

# Preprocess the data
X = combria_data.drop(columns=['Classification'])
y = combria_data['Classification']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    # Add more features as necessary

    # Prepare input data for prediction
    input_data = [[feature1, feature2]]  # Add more features as necessary
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    logreg_prediction = logreg.predict(input_data_scaled)
    rf_prediction = rf.predict(input_data_scaled)
    svm_prediction = svm.predict(input_data_scaled)

    # Choose one prediction to display (e.g., Logistic Regression)
    prediction = logreg_prediction[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
