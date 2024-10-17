from functions import *

def train(sumcontrols):
    # Load the data from the CSV file
    df = pd.read_csv('Datasets/Afghanistan2017districts_neighbors.csv')

    # Select features and target variable EXCLUDING 'sum_controls'
    features = []
    if(sumcontrols):
        features = ['count_gdset', 'count_terrorism','sum_controls']
    else:
        features = ['count_gdset', 'count_terrorism']
    target = 'control'
    X = df[features]
    y = df[target]

    # Split the data into training aboolnd testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choose and instantiate the model - Random Forest is a good starting point
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    model_filename = 'Models/trainedmodel.joblib'
    if(sumcontrols):
        model_filename = 'Models/trained_with_sums_model.joblib'
    joblib.dump(model, model_filename)

    print(f"Model saved to {model_filename}")

    scaler_filename = 'Models/trained_scaler.joblib'
    if(sumcontrols):
        scaler_filename = 'Models/trained_with_sums_scaler.joblib'
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report for more detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

train(sumcontrols=True)
train(sumcontrols=False)