from functions import *

# ... (kod wczytywania danych gdset_march_2017 i terrorism_march_2017 z marca 2017) ...
# Load the datasets
gdset_afghanistan = pd.read_csv('Datasets/gdset_afghanistan.csv')
terrorism_afghanistan = pd.read_csv('Datasets/terrorism_afghanistan_withadm.csv')

# Filter data for March 2017
gdset_march_2017 = gdset_afghanistan[
    (pd.to_datetime(gdset_afghanistan['date_start']).dt.year == 2017)
    & (pd.to_datetime(gdset_afghanistan['date_start']).dt.month == 3)
]

terrorism_march_2017 = terrorism_afghanistan[
    (terrorism_afghanistan['iyear'] == 2017)
    & (terrorism_afghanistan['imonth'] == 3)
]

# Load the saved model
model_filename = 'trained_control_model.joblib'
loaded_model = joblib.load(model_filename)

# Prepare the data for prediction 
# Assuming you have dataframes 'gdset_march_2017' and 'terrorism_march_2017' 
# for March 2017, group them by district and count incidents:

grouped_march_2017 = gdset_march_2017.groupby('adm_2').agg({
    'latitude': 'mean', 
    'longitude': 'mean'
}).reset_index()

grouped_march_2017['count_gdset'] = grouped_march_2017['adm_2'].apply(lambda x: len(gdset_march_2017[gdset_march_2017['adm_2'] == x]))
grouped_march_2017['count_terrorism'] = grouped_march_2017['adm_2'].apply(lambda x: len(terrorism_march_2017[terrorism_march_2017['adm_2'] == x]))

# Select the features for prediction
X_new = grouped_march_2017[['count_gdset', 'count_terrorism']] 

# Feature scaling - IMPORTANT: Use the same scaler fitted on training data
scaler_filename = 'trained_control_scaler.joblib'
scaler = joblib.load(scaler_filename)


X_new = scaler.transform(X_new) # Assuming 'scaler' is the same scaler from training

# Predict control using the loaded model
predicted_control = loaded_model.predict(X_new)

# Add the predicted control to the DataFrame
grouped_march_2017['predicted_control'] = predicted_control

# Print or save the results
print(grouped_march_2017[['adm_2', 'predicted_control']]) 