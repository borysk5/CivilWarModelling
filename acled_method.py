from functions import *

def calculate_control(events_df):
    """
    Calculates control of districts in Afghanistan based on ACLED methodology.

    Args:
        events_df (pd.DataFrame): DataFrame containing ACLED events data.

    Returns:
        pd.DataFrame: DataFrame with district_name, control, active, latitude, and longitude columns.
    """

    # Define relevant event types and sub-event types
    territorial_takeover_events = [
        "Non-state actor overtakes territory",
        "Government regains territory",
        "Non-violent transfer of territory"
    ]
    dominance_events = [
        "Non-state actor overtakes territory",
        "Government regains territory",
        "Armed clash",
        "Change to group/activity",
        "Non-violent transfer of territory",
        "Chemical weapon",
        "Air/drone strike",
        "Suicide bomb",
        "Shelling/artillery/missile attack",
        "Remote explosive/landmine/IED",
        "Grenade"
    ]
    activity_events = dominance_events + ["Headquarters or base established", "Disrupted weapons use"]

    # Group events by quarter and district
    events_df['quarter'] = events_df['event_date'].dt.to_period('Q')
    events_by_district_quarter = events_df.groupby(['admin2', 'quarter'])

    control_data = {}
    for (district, quarter), events in events_by_district_quarter:
        # Calculate control
        control = determine_control(events, territorial_takeover_events, dominance_events)

        # Calculate activity
        active = determine_activity(events, activity_events)

        # Get the latitude and longitude (taking the mean for this district and quarter)
        latitude = events['latitude'].mean()
        longitude = events['longitude'].mean()

        control_data[district] = {
            'district_name': district, 
            'control': control, 
            'active': active,
            'latitude': latitude,
            'longitude': longitude
        }

    return pd.DataFrame(control_data.values())


def determine_control(events, territorial_takeover_events, dominance_events):
    """
    Determines control of a district based on ACLED methodology.

    Args:
        events (pd.DataFrame): DataFrame containing events for a specific district and quarter.
        territorial_takeover_events (list): List of event sub-event types that indicate territorial takeover.
        dominance_events (list): List of event sub-event types that indicate dominance.

    Returns:
        str: Actor in control or "contested".
    """

    # 1. Takeover
    takeovers_by_actor = events[events['sub_event_type'].isin(territorial_takeover_events)]['actor1'].value_counts()
    if len(takeovers_by_actor) > 0 and takeovers_by_actor.iloc[0] >= 2 * takeovers_by_actor.iloc[1:].sum():
        return takeovers_by_actor.index[0]

    # 2. Dominance
    dominance_by_actor = events[events['sub_event_type'].isin(dominance_events)]['actor1'].value_counts()
    if len(dominance_by_actor) > 0 and dominance_by_actor.iloc[0] >= (2/3) * dominance_by_actor.sum():
        return dominance_by_actor.index[0]

    # 3. Historical Control - This requires additional data and logic not provided in the document.
    # You'll need to implement this based on your understanding of historical control.

    return "contested"

def determine_activity(events, activity_events):
    """
    Determines if a district is active based on event count.

    Args:
        events (pd.DataFrame): DataFrame containing events for a specific district and quarter.
        activity_events (list): List of event sub-event types that indicate activity.

    Returns:
        int: 1 if active, 0 if inactive.
    """
    return 1 if len(events[events['sub_event_type'].isin(activity_events)]) >= 10 else 0


def calculate():
# Load events data
    events_df = pd.read_csv("Datasets/acled_afghanistan.csv", parse_dates=['event_date'])

    # 1. Limit data to March 2017
    events_df = events_df[events_df['event_date'] < pd.to_datetime("2017-04-01")]

    # 1. Remove records with "Unidentified Armed Group (Afghanistan)"
    events_df = events_df[events_df['actor1'] != "Unidentified Armed Group (Afghanistan)"]
    events_df = events_df[events_df['actor2'] != "Unidentified Armed Group (Afghanistan)"]

    # 2. Combine police and military forces into "Military Forces of Afghanistan (2014-2021)"
    police_forces = ["Police Forces of Afghanistan (2014-2021)", 
                    "Police Forces of Afghanistan (2014-2021) National Directorate of Security","Militia (Pro-Government)","Military Forces of Afghanistan (2014-2021) Special Forces"]
    events_df['actor1'] = events_df['actor1'].replace(police_forces, "Military Forces of Afghanistan (2014-2021)")
    events_df['actor2'] = events_df['actor2'].replace(police_forces, "Military Forces of Afghanistan (2014-2021)")

    # Calculate control for each district
    result_df = calculate_control(events_df)

    result_df.to_csv("DataResults/acled_result.csv", index=False)

    # 2 & 3. Create Folium map
    afghanistan_map = folium.Map(location=[33.93911, 67.709953], zoom_start=6)

    color_mapping = {
        "contested": "yellow",
        "Taliban": "green",
        "Military Forces of Afghanistan (2014-2021)": "red",
        "other": "blue" 
    }
    image = Image.open("Afghanistanblank.png")

    for index, row in result_df.iterrows():
        if row['control'] in color_mapping:
            color = color_mapping[row['control']]
        else:
            color = color_mapping['other']


        control = row['control']
        active = row['active']

        colour=(255,255,255)
        if(control=='contested'):
            colour=(255,255,0)
        elif(control=='Taliban') and active==1:
            colour=(150,150,150)
        elif(control=='Taliban') and active==0:
            colour=(40,40,40)
        elif(control=='Military Forces of Afghanistan (2014-2021)') and active==1:
            colour=(255,165,0)
        elif(control=='Military Forces of Afghanistan (2014-2021)') and active==0:
            colour=(255,0,0)

        x,y = coordtopx(row['latitude'], row['longitude'])
        bucket_fill(image,x,y,colour)



        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1
        ).add_to(afghanistan_map)

    # Save the map
    image.save("ImageOutputs/afghanistan_acled.png")
    afghanistan_map.save("Maps/afghanistan_acled.html")

    acled_control = []
    reference_control = []


    for index, row in result_df.iterrows():
        if read_color(row['latitude'], row['longitude']) != 'U':  # Exclude districts with undetermined control
            control = row['control']
            active = row['active']
            if(control=='contested'):
                acled_control.append('D')
            elif(control=='Taliban') and active==1:
                acled_control.append('DR')
            elif(control=='Taliban') and active==0:
                acled_control.append('R')
            elif(control=='Military Forces of Afghanistan (2014-2021)') and active==1:
                acled_control.append('DG')
            elif(control=='Military Forces of Afghanistan (2014-2021)') and active==0:
                acled_control.append('G')
            else:
                continue
            reference_control.append(read_color(row['latitude'], row['longitude']))

    # Create agreement table (confusion matrix)
    agreement_table = pd.crosstab(
        pd.Series(acled_control, name='ACLED Control'),
        pd.Series(reference_control, name='Reference Control')
    )

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(
        pd.factorize(acled_control)[0],
        pd.factorize(reference_control)[0]
    )

    # Calculate accuracy
    total_compared = len(acled_control)
    correct_predictions = sum(1 for i in range(total_compared) if acled_control[i] == reference_control[i])
    accuracy = correct_predictions / total_compared

    # Save statistics to a text file
    with open("DataResults/acled_correlation.txt", "w") as f:
        f.write("Agreement Table:\n")
        f.write(agreement_table.to_string())
        f.write("\n\n")
        f.write(f"Pearson Correlation: {correlation:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

calculate()