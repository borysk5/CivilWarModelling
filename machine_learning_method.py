from functions import *

# Load datasets (make sure they cover the period from January 2003)
gdset_afghanistan = pd.read_csv('Datasets/gdset_afghanistan.csv')
terrorism_afghanistan = pd.read_csv('Datasets/terrorism_afghanistan_withadm.csv')

color_mapRGB = {
    4: (255,255,0),      # Rebel Control
    3: (150,150,150),   # Disputed, closer to Rebel
    2: (255,255,0),    # Highly Disputed
    1: (255,165,0),  # Disputed, closer to Government
    0: (255,0,0)      # Government Control
}

def calculate():
# Load the saved model and scaler
    model_filename = 'Models/trainedmodel.joblib'
    scaler_filename = 'Models/trained_scaler.joblib'
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)

    district_data = gdset_afghanistan.groupby('adm_2').agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    district_data = district_data.set_index('adm_2').to_dict('index')


    control_dict = {}
    for district in district_data:
        control_dict[district] = {
            'control': 0, 
            'latitude': district_data[district]['latitude'],
            'longitude': district_data[district]['longitude']
        }


    end_year = 2017
    end_month = 3

    current_year = 2017
    current_month = 3

    print(f"Processing: Year {current_year}, Month {current_month}") 

    # Filter DataFrames for the current month and year
    gdset_current_month = gdset_afghanistan[
        (pd.to_datetime(gdset_afghanistan['date_start']).dt.year == current_year)
        & (pd.to_datetime(gdset_afghanistan['date_start']).dt.month == current_month)
    ]
    
    terrorism_current_month = terrorism_afghanistan[
        (terrorism_afghanistan['iyear'] == current_year)
        & (terrorism_afghanistan['imonth'] == current_month)
    ]
    
    # Create a copy of the control dictionary to store previous month's data
    previous_month_control = control_dict.copy() 

    for district in control_dict:
        # Count events in the current district
        count_gdset = len(gdset_current_month[gdset_current_month['adm_2'] == district])
        count_terrorism = len(terrorism_current_month[terrorism_current_month['adm_2'] == district])

        # Prepare data for prediction
        features = pd.DataFrame([[count_gdset, count_terrorism]], 
                                columns=['count_gdset', 'count_terrorism'])
        features = loaded_scaler.transform(features)

        # Predict control
        predicted_control = loaded_model.predict(features)[0]

        # Update the control dictionary
        control_dict[district]['control'] = predicted_control

    # Move to the next month
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1

    # Save the March 2017 control data to a CSV file
    march_2017_control = pd.DataFrame.from_dict(control_dict, orient='index')

    march_2017_control.to_csv("DataResults/machine_learning_2017.csv", index=True)
    machine_correctness(False)

def calculate_sumcontrols():
# Load the saved model and scaler
    model_filename = 'Models/trained_with_sums_model.joblib'
    scaler_filename = 'Models/trained_with_sums_scaler.joblib'
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)

    district_data = gdset_afghanistan.groupby('adm_2').agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    district_data = district_data.set_index('adm_2').to_dict('index')

    for district in district_data:
        district_data[district]['neighbors'] = find_closest_districts(district, district_data)

    control_dict = {}
    for district in district_data:
        control_dict[district] = {
            'control': 0, 
            'latitude': district_data[district]['latitude'],
            'longitude': district_data[district]['longitude'],
            'neighbors': district_data[district]['neighbors']
        }

    # Function to get district control from previous month (with handling for first month)
    def get_previous_control(district, previous_month_control):
        try:
            return previous_month_control[district]['control']
        except KeyError:
            return 0  # Assume government control for the first month 

    # Iterate through months from January 2003 to March 2017
    start_year = 2003
    start_month = 1
    end_year = 2017
    end_month = 3

    current_year = start_year
    current_month = start_month

    while current_year < end_year or (current_year == end_year and current_month <= end_month):
        print(f"Processing: Year {current_year}, Month {current_month}") 

        # Filter DataFrames for the current month and year
        gdset_current_month = gdset_afghanistan[
            (pd.to_datetime(gdset_afghanistan['date_start']).dt.year == current_year)
            & (pd.to_datetime(gdset_afghanistan['date_start']).dt.month == current_month)
        ]
        
        terrorism_current_month = terrorism_afghanistan[
            (terrorism_afghanistan['iyear'] == current_year)
            & (terrorism_afghanistan['imonth'] == current_month)
        ]
        
        # Create a copy of the control dictionary to store previous month's data
        previous_month_control = control_dict.copy() 

        for district in control_dict:
            # Count events in the current district
            count_gdset = len(gdset_current_month[gdset_current_month['adm_2'] == district])
            count_terrorism = len(terrorism_current_month[terrorism_current_month['adm_2'] == district])

            # Find the two closest districts and get their control status from the previous month
            sum_controls = 0
            for _, neighbor in control_dict[district]['neighbors']:
                sum_controls += get_previous_control(neighbor, previous_month_control)

            # Prepare data for prediction

            features = pd.DataFrame([[count_gdset, count_terrorism, sum_controls]], 
                                        columns=['count_gdset', 'count_terrorism', 'sum_controls'])
            features = loaded_scaler.transform(features)
            # Predict control
            predicted_control = loaded_model.predict(features)[0]

            # Update the control dictionary
            control_dict[district]['control'] = predicted_control

        # Move to the next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    # Save the March 2017 control data to a CSV file
    march_2017_control = pd.DataFrame.from_dict(control_dict, orient='index')
    march_2017_control.to_csv("DataResults/machine_learning_2017_sumcontrols.csv", index=True)
    print("March 2017 control data saved to machine_learning_2017.csv")
    machine_correctness(True)


calculate_sumcontrols()