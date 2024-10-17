from functions import *

states = ['R', 'DR', 'D', 'DG', 'G']  # Rebel, Disputed (closer to R/G), Government
n_states = len(states)

transition_matrix = np.array([
    [0.25, 0.50, 0.03, 0.20, 0.02], # R
    [0.25, 0.15, 0.08, 0.50, 0.02], # DR
    [0.05, 0.03, 0.05, 0.85, 0.02], # D
    [0.03, 0.08, 0.15, 0.13, 0.62], # DG
    [0.05, 0.08, 0.47, 0.03, 0.37] # G
])

# Emission Probabilities (heuristic, can be refined)
emission_matrix = np.array([
    [0.60, 0.18, 0.17, 0.05],  # R
    [0.05, 0.60, 0.18, 0.17],  # DR
    [0.05, 0.18, 0.60, 0.17],  # D
    [0.05, 0.18, 0.17, 0.60],  # DG
    [0.60, 0.05, 0.18, 0.17]   # G
])

# Load the datasets
gdset_afghanistan = pd.read_csv('Datasets/gdset_afghanistan.csv')
terrorism_afghanistan = pd.read_csv('Datasets/terrorism_afghanistan_withadm.csv')

# Filter data for relevant period
gdset_afghanistan['date_start'] = pd.to_datetime(gdset_afghanistan['date_start'])
gdset_afghanistan['date_end'] = pd.to_datetime(gdset_afghanistan['date_end'])

# Aggregate data by administrative region (adm_2)
grouped = gdset_afghanistan.groupby('adm_2').agg({
    'latitude': 'mean',
    'longitude': 'mean'
}).reset_index()

# Initialize control for each district
grouped_dict = grouped.set_index('adm_2').T.to_dict()
for key in grouped_dict:
    grouped_dict[key]['control'] = 'G'  # Assuming initial control is Government

# Function to determine control based on previous control and emissions
def control_basedon_previous_control_and_emissions(previous_control, emission_type):
    """
    Estimate control based on previous month's control and current emission type.

    Args:
        previous_control (str): Control state in the previous month ('R', 'DR', 'D', 'DG', 'G').
        emission_type (int): Type of emission observed (0-3, as defined in the paper).

    Returns:
        str: Most likely control state for the current month.
    """
    previous_state_index = states.index(previous_control)
    probabilities = [
        transition_matrix[previous_state_index][i] * emission_matrix[i][emission_type]
        for i in range(n_states)
    ]
    most_likely_state_index = np.argmax(probabilities)
    return states[most_likely_state_index]


start_year = 2003
start_month = 1
end_year = 2017
end_month = 3

current_year = start_year
current_month = start_month

while current_year < end_year or (current_year == end_year and current_month <= end_month):
    # Filter DataFrames for the current month and year
    gdset_current_month = gdset_afghanistan[
        (gdset_afghanistan['date_start'].dt.year == current_year)
        & (gdset_afghanistan['date_start'].dt.month == current_month)
    ]
    
    terrorism_current_month = terrorism_afghanistan[
        (terrorism_afghanistan['iyear'] == current_year)
        & (terrorism_afghanistan['imonth'] == current_month)
    ]
    
    for name in grouped['adm_2']:  # Iterate over district names
        count_gdset = len(gdset_current_month[gdset_current_month['adm_2'] == name])
        count_terrorism = len(terrorism_current_month[terrorism_current_month['adm_2'] == name])
        previous_month_control = grouped_dict[name]['control']
        
        # Determine emission type based on event counts (example logic)
        if count_gdset == 0 and count_terrorism == 0:
            emission_type = 0  # No events
        elif count_gdset > count_terrorism:
            emission_type = 1  # More conventional fighting
        elif count_terrorism > count_gdset:
            emission_type = 3  # More terrorist attacks
        else:
            emission_type = 2  # Similar levels of both

        # Update control using the defined function
        grouped_dict[name]['control'] = control_basedon_previous_control_and_emissions(previous_month_control, emission_type)

    # Print progress for each month
    print(f"Year: {current_year}, Month: {current_month} - Data Processed")
    
    # Update month and year for the next iteration
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1

# Create a map centered on Afghanistan
afghanistan_map = folium.Map(location=[33.9391, 67.7100], zoom_start=6)

# Define a color mapping for control states
color_map = {
    'R': 'red',      # Rebel Control
    'DR': 'orange',   # Disputed, closer to Rebel
    'D': 'yellow',    # Highly Disputed
    'DG': 'lightgreen',  # Disputed, closer to Government
    'G': 'green'      # Government Control
}

color_mapRGB = {
    'R': (255,255,0),      # Rebel Control
    'DR': (150,150,150),   # Disputed, closer to Rebel
    'D': (255,255,0),    # Highly Disputed
    'DG': (255,165,0),  # Disputed, closer to Government
    'G': (255,0,0)      # Government Control
}

image = Image.open("Afghanistanblank.png")
# Add markers to the map for each district and its estimated control
for district, data in grouped_dict.items():
    folium.CircleMarker(
        location=[data['latitude'], data['longitude']],
        radius=5,
        color=color_map[data['control']],
        fill=True,
        fill_color=color_map[data['control']],
        fill_opacity=0.7,
        popup=f"District: {district}<br>Control: {data['control']}"
    ).add_to(afghanistan_map)

    x,y = coordtopx(data['latitude'], data['longitude'])
    bucket_fill(image,x,y,color_mapRGB[data['control']])



image.save("ImageOutputs/gdemethod_custom_afghanistan.png")
# Save the map to an HTML file
afghanistan_map.save("Maps/gdemethod_custom_afghanistan.html") 

hmm_control = []
reference_control = []


for district, data in grouped_dict.items():
    if data['control'] != 'U':  # Exclude districts with undetermined control
        hmm_control.append(data['control'])
        reference_control.append(read_color(data['latitude'], data['longitude']))

# Create agreement table (confusion matrix)
agreement_table = pd.crosstab(
    pd.Series(hmm_control, name='HMM Control'),
    pd.Series(reference_control, name='Reference Control')
)

order = ['G', 'DG', 'D', 'DR', 'R']
agreement_table = agreement_table.reindex(index=order, columns=order)

# Calculate Pearson correlation
correlation, p_value = pearsonr(
    pd.factorize(hmm_control)[0],
    pd.factorize(reference_control)[0]
)

# Calculate accuracy
total_compared = len(hmm_control)
correct_predictions = sum(1 for i in range(total_compared) if hmm_control[i] == reference_control[i])
accuracy = correct_predictions / total_compared

# Save statistics to a text file
with open("DataResults/gde_custom_statistics.txt", "w") as f:
    f.write("Agreement Table:\n")
    f.write(agreement_table.to_string())
    f.write("\n\n")
    f.write(f"Pearson Correlation: {correlation:.4f}\n")
    f.write(f"P-value: {p_value:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")

print("Statistics saved to gde_statistics.txt")