import csv
import folium
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import json
from geopy.geocoders import Nominatim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Don't forget to import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Function to find the two closest districts (reuse from previous examples)
def find_closest_districts(target_district, district_data):
    distances = []
    for name, data in district_data.items():
        if name != target_district:
            lat_diff = abs(district_data[target_district]['latitude'] - data['latitude'])
            lon_diff = abs(district_data[target_district]['longitude'] - data['longitude'])
            distance = lat_diff + lon_diff  # Using simple sum of differences
            distances.append((distance, name))
    distances.sort()  # Sort based on distance
    return distances[:2]  # Return the two closest

def machine_correctness(sumcontrols):
    color_mapRGB = {
    4: (255,255,0),      # Rebel Control
    3: (150,150,150),   # Disputed, closer to Rebel
    2: (255,255,0),    # Highly Disputed
    1: (255,165,0),  # Disputed, closer to Government
    0: (255,0,0)      # Government Control
    }

# Load the predicted control data
    df_predicted = pd.read_csv("DataResults/machine_learning_2017_sumcontrols.csv")
    if(not sumcontrols):
        df_predicted = pd.read_csv("DataResults/machine_learning_2017.csv")

    # Create lists to store control data for comparison
    ml_control = []
    reference_control = []

    # Control mapping for numerical conversion
    control_mapping = {'G': 0, 'DG': 1, 'D': 2, 'DR': 3, 'R': 4}
    image = Image.open("Afghanistanblank.png")
    # Iterate through the predicted data and compare
    for index, row in df_predicted.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        predicted_control = row['control']
        x,y = coordtopx(latitude, longitude)
        bucket_fill(image,x,y,color_mapRGB[predicted_control])
        # Get the reference control using read_color function
        ref_control = read_color(latitude, longitude)

        # Only consider districts with determined reference control
        if ref_control != 'U':
            ml_control.append(predicted_control)
            reference_control.append(control_mapping[ref_control])
    if(sumcontrols):
        image.save("ImageOutputs/machine_learning_sumcontrols.png")
    else:
        image.save("ImageOutputs/machine_learning.png")
    # Create agreement table (confusion matrix)
    agreement_table = pd.crosstab(
        pd.Series(ml_control, name='ML Control'),
        pd.Series(reference_control, name='Reference Control')
    )

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(ml_control, reference_control)

    # Calculate accuracy
    total_compared = len(ml_control)
    correct_predictions = sum(1 for i in range(total_compared) if ml_control[i] == reference_control[i])
    accuracy = correct_predictions / total_compared

    # Save statistics to a text file
    file_name = "DataResults/machine_statistics.txt"
    if(sumcontrols):
        file_name = "DataResults/machine_statistics_sumcontrols.txt"
    with open(file_name, "w") as f:
        f.write("Agreement Table:\n")
        f.write(agreement_table.to_string())
        f.write("\n\n")
        f.write(f"Pearson Correlation: {correlation:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    print("Statistics saved to machine_statistics_mydata.txt")

def coordtopx(latitude, longitude, image_size=(1314, 1002), map_coords=((29.3218, 60.4053), (38.4687, 74.8922))):
    """
    Convert latitude and longitude coordinates to pixel coordinates on a map image.
    
    Args:
    - latitude: Latitude coordinate
    - longitude: Longitude coordinate
    - image_size: Tuple containing the size of the map image in pixels (default: (1314, 1002))
    - map_coords: Tuple containing the coordinates of the map corners ((min_lat, min_long), (max_lat, max_long))
    
    Returns:
    - Tuple representing the pixel coordinates (x, y)
    """
    min_lat, min_long = map_coords[0]
    max_lat, max_long = map_coords[1]
    
    # Calculate the pixel coordinates based on the given latitude and longitude
    x = int((longitude - min_long) / (max_long - min_long) * image_size[0]) #poziom
    #print(x)
    # Adjust y coordinate to account for reversed y-axis direction
    y = int(image_size[1] - (latitude - min_lat) / (max_lat - min_lat) * image_size[1]) #pion
    
    return x, y





def get_average_color(image, x, y, radius=1):
    """
    Get the average color of pixels around a given point in an image.
    
    Args:
    - image: PIL Image object
    - x: x-coordinate of the center point
    - y: y-coordinate of the center point
    - radius: radius around the center point to consider (default: 1)
    
    Returns:
    - Tuple representing the average color (R, G, B)
    """
    pixels = image.load()
    width, height = image.size
    colors = []
    
    for i in range(max(0, x - radius), min(width, x + radius + 1)):
        for j in range(max(0, y - radius), min(height, y + radius + 1)):
            if(pixels[i, j]==(255,255,255)):
                continue
            colors.append(pixels[i, j])
    
    avg_color = tuple(sum(channel) / len(colors) for channel in zip(*colors))
    return avg_color

#image = Image.open("Af.png")
#j = coordtopx(37.3385,65.7367)
#print( get_average_color(image,j[0],j[1]) )

import ast
import math

def euclidean_distance(color1, color2):
    """
    Calculate the Euclidean distance between two colors.
    
    Args:
    - color1: Tuple representing the first color (R, G, B)
    - color2: Tuple representing the second color (R, G, B)
    
    Returns:
    - Euclidean distance between the two colors
    """
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

def find_closest_color(color, color_palette):
    """
    Find the closest predefined color to the given color.
    
    Args:
    - color: Tuple representing the target color (R, G, B)
    - color_palette: Dictionary containing predefined colors and their RGB values
    
    Returns:
    - Name of the closest predefined color
    """
    closest_color = None
    min_distance = float('inf')
    
    for name, rgb in color_palette.items():
        distance = euclidean_distance(color, rgb)
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color

# Define the list of predefined colors
color_palette = {
    'black': (0, 0, 0),
    'yellow': (249, 247, 166),
    'orange': (255, 202, 0),
    'red': (255, 0, 0),
    'green': (55, 255, 99),
    'grey': (248, 242, 218)
}

# Read the content of the txt file and process each row
def do():
    with open("result.txt", "r") as file:
        lines = file.readlines()



    with open("result_with_colors.txt", "w") as output_file:
        for line in lines:
            # Split the line into location and color tuple
            location, color_tuple = line.split(": ")
            location = location.strip()
            
            # Convert the color tuple string to a tuple of integers
            color = ast.literal_eval(color_tuple)
            
            # Find the closest predefined color
            closest_color = find_closest_color(color, color_palette)
            
            # Write the location with the closest color to the output file
            output_file.write(f"{location}: {closest_color}\n")

import folium
from folium.plugins import MarkerCluster

def poisson_pmf(k, lambd):
    """Calculate the probability mass function of a Poisson distribution."""
    return (lambd ** k) * math.exp(-lambd) / math.factorial(k)

colors_dict = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3,
    "black": 4
}

colors_dict_rev = ["green","yellow","orange","red","black"]


x1=[2.714285714,	0.02857142857,	8.771428571,	0.02857142857,	0.9428571429]
x2=[1.831460674,	0.03370786517,	9.719101124,	0.05617977528,	0.606741573]
x3=[1.983870968,	0.04838709677,	6.483870968,	0.01612903226,	0.2580645161]
x4=[1.904761905,	0.03174603175,	4.952380952,	0.06349206349,	0.3650793651]
x5=[0.7640449438,	0.02247191011,	2.247191011,	0.01123595506,	0.202247191]
x_list =[x5,x4,x3,x2,x1]


percentages = [
    [90, 5, 2, 1, 0],
    [17, 60, 17, 5, 1],
    [3, 17, 60, 17, 3],
    [1, 5, 17, 60, 17],
    [0, 1, 2, 5, 90]
]



def recursion(color, incidents):
    number = colors_dict[color]
    probs = [0,0,0,0,0]
    for x in [0,1,2,3,4]: #government->rebel
        f = 1
        for xy in [0,1,2,3,4]:
            f *= poisson_pmf(incidents[xy],x_list[x][xy])
        f*=percentages[number][x]
        probs[x] = f
    return colors_dict_rev[probs.index(max(probs))]      

#zx = recursion("black",[2,0,8,10,0]) #Reb. explosion,	Gov. Violence,	Armed clash,	Rebel violence,	Gov. explosion

def read_color(latitude,longitude):
    image = Image.open("DataMaps/afghanistan2017.png")
    pixel_coords = coordtopx(latitude, longitude)
    avg_color = get_average_color(image, *pixel_coords, radius=1)
    #color = ast.literal_eval(avg_color)
    
    # Find the closest predefined color
    color_palette2 = {
'R': (0,0,0),
'DR': (220,68,53),
'D': (232,193,74),
'DG': (249,247,166),
'G': (118,170,91),
'U': (255,255,255)
}
    closest_color = find_closest_color(avg_color, color_palette2)
    return closest_color

def bucket_fill(image, x, y, replacement_color):
    """
    Perform bucket fill operation on the image starting from coordinates (x, y)
    with the target_color, replacing it with replacement_color.
    """
    width, height = image.size
    target_color = image.getpixel((x, y))
    if(target_color!=(216,217,218,255)):
        return
    stack = [(x, y)]
    while stack:
        current_x, current_y = stack.pop()
        if (0 <= current_x < width and 0 <= current_y < height and
            image.getpixel((current_x, current_y)) == target_color):
            image.putpixel((current_x, current_y), replacement_color)
            stack.extend([(current_x+1, current_y), (current_x-1, current_y),
                          (current_x, current_y+1), (current_x, current_y-1)])
    
def myfunction_acled(file_path):
    #clear_file("entry.txt")
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = pd.read_csv(csvfile)
        #csvreader = csvreader[csvreader['geo_precision'] < 2]
        distinct_admin2 = csvreader['admin2'].unique()
        admin2_dict = {value: {'active': 0,'control': "gov", 'date':0,'latitude': csvreader.loc[(csvreader['admin2'] == value) & (csvreader['geo_precision'] == 2), 'latitude'].iloc[0], 'longitude': csvreader.loc[(csvreader['admin2'] == value) & (csvreader['geo_precision'] == 2), 'longitude'].iloc[0]} for value in distinct_admin2}
        csvreader = csvreader[(csvreader['event_type'] == 'Battles')|(csvreader['event_type']=='Explosions/Remote violence')|(csvreader['event_type']=='Strategic developments')]
        start_date = "2020-01-01"
        end_date = "2021-05-01"
        date_range = pd.date_range(start=start_date, end=end_date, freq='3M')
        sides=['Taliban','Islamic State (Afghanistan)']
        for i in range(len(date_range) - 1):
            period_start = date_range[i]
            period_end = date_range[i + 1]
            monthly_df = csvreader[(csvreader['event_date'] >= period_start.strftime("%Y-%m-%d")) & (csvreader['event_date'] < period_end.strftime("%Y-%m-%d"))]
            count_per_admin2 = monthly_df.groupby('admin2').size()
            average_per_admin2 = count_per_admin2.mean()
            std_dev_per_admin2 = count_per_admin2.std()
            minimum = average_per_admin2 - std_dev_per_admin2


            for key in monthly_df['admin2'].unique():
                    md_df = monthly_df[monthly_df['admin2'] == key]
                    ae = len(md_df)
                    if(ae < 5):
                        admin2_dict[key]['active'] = 0
                    else:
                        admin2_dict[key]['active'] = 1
                    if(admin2_dict[key]['control']=='contested' or ae >= minimum):
                        gov_takeovers = md_df[(md_df['inter1'] == 1) & ((md_df['sub_event_type'] == 'Headquarters or base established') | (md_df['sub_event_type'] == 'Government regains territory') | (md_df['sub_event_type'] == 'Non-violent transfer of territory'))].shape[0]
                        talib_takeovers = md_df[(md_df['actor1'] == 'Taliban') & ((md_df['sub_event_type'] == 'Headquarters or base established') | (md_df['sub_event_type'] == 'Non-state actor overtakes territory') | (md_df['sub_event_type'] == 'Non-violent transfer of territory'))].shape[0]
                        isis_takovers = md_df[(md_df['actor1'] == 'Islamic State (Afghanistan)') & ((md_df['sub_event_type'] == 'Headquarters or base established') | (md_df['sub_event_type'] == 'Non-state actor overtakes territory') | (md_df['sub_event_type'] == 'Non-violent transfer of territory'))].shape[0]
                        takeovers = gov_takeovers + talib_takeovers + isis_takovers
                        if(takeovers>0 & 3*gov_takeovers >= 2*takeovers):
                            admin2_dict[key]['control'] = 'gov'
                        elif(takeovers>0 & 3*talib_takeovers >= 2* takeovers):
                            admin2_dict[key]['control'] = 'tal'
                        elif(takeovers>0 & 3*isis_takovers >= 2* takeovers):
                            admin2_dict[key]['control'] = 'isis'
                        else:
                            mx_df = md_df
                            t0 = len(mx_df[(mx_df['inter1'] == 1)])
                            t1 = len(mx_df[(mx_df['actor1'] == 'Taliban')])
                            t2 = len(mx_df[(mx_df['actor1'] == 'Islamic State (Afghanistan)')])
                            ev=len(mx_df)
                            if(ev >= 1):
                                if(t0>=2/3*ev and t0>t1):
                                    admin2_dict[key]['control'] = 'gov'
                                elif(t1>=2/3*ev and t1>t0):
                                    admin2_dict[key]['control'] = 'tal'
                                elif(t2>=2/3*ev and t2>t0 and t2>t1):
                                    admin2_dict[key]['control'] = 'isis'
                                elif(ev >= 5):
                                    admin2_dict[key]['control'] = 'contested'

        m = folium.Map(location=[28.0, 29.0], zoom_start=6)
        image = Image.open("Afghanistanblank.png")

        for key, value in admin2_dict.items():
            # Extract color and other necessary values from the dictionary
            zh=1
            control = value['control']
            active = value['active']
            if(control=='gov'):
                if(active==0):
                    zh=1
                else:
                    zh=2
                color = (255,0,0)
            elif(control=='tal'):
                if(active==0):
                    zh=5
                else:
                    zh=4
                color=(200,200,200)
            elif(control=='isis'):
                color=(100,100,100)
            elif control=='contested':
                color = (255,255,0)
                zh=3

            latitude = value['latitude']
            longitude = value['longitude']
            with open("correlation.txt", "a") as myfile:
                myfile.write(key + ' ' + str(zh) + ' ' + str(read_color(latitude,longitude))+'\n')
            
            #color = color_palette[value['control']]
            latitude = value['latitude']
            longitude = value['longitude']

            if color == 0:
                color = (255, 255, 255)
            (r, g, b) = color
            
            x,y = coordtopx(latitude, longitude)
            bucket_fill(image,x,y,color)
            # Create a folium Popup
            popup = folium.Popup(key, parse_html=True)
            
            # Create a CircleMarker and add it to the map
            if (r, g, b) != (255, 255, 255):
                folium.CircleMarker(location=[latitude, longitude],
                                    radius=5,
                                    color='#{:02x}{:02x}{:02x}'.format(0, 0, 0),
                                    fill=True,
                                    fill_opacity=1,
                                    fill_color='#{:02x}{:02x}{:02x}'.format(r, g, b),
                                    popup=popup).add_to(m)

        
        
        image.save("afghanistanmapped.png")
        m.save("afg_full2.html")

#myfunction_acled("pakistan.csv")

def myfunction_csv(file_path):
    #clear_file("entry.txt")
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = pd.read_csv(csvfile)
        #csvreader2 = pd.read_csv('ambazoniaterror.csv')
        #csvreader3 = pd.read_csv('outambazonia.csv')
        #csvreader = pd.merge(csvreader, csvreader2, how='outer')
        #csvreader = pd.merge(csvreader, csvreader3, how='outer')
        #csvreader = csvreader[csvreader['event_date'] > '2018-03-14']
        
        #csvreader = csvreader[csvreader['geo_precision'] < 2]

        #csvreader = pd.concat([csvreader, df], ignore_index=True, sort=False)
        #csvreader = csvreader[csvreader['event_date'] < '2021-05-10']


        # Print distinct values of "admin2" column
        distinct_admin2 = csvreader['admin2'].unique()


# Create a dictionary with distinct values of "admin2" column as keys and dictionary as values
        admin2_dict = {value: {'control': "green", 'date':0,'latitude': csvreader.loc[(csvreader['admin2'] == value) & (csvreader['geo_precision'] == 2), 'latitude'].iloc[0], 'longitude': csvreader.loc[(csvreader['admin2'] == value) & (csvreader['geo_precision'] == 2), 'longitude'].iloc[0]} for value in distinct_admin2}

        start_date = "2003-01-01"
        end_date = "2021-05-01"
        date_range = pd.date_range(start=start_date, end=end_date, freq='3M')
        for i in range(len(date_range) - 1):
            period_start = date_range[i]
            period_end = date_range[i + 1]
            monthly_df = csvreader[(csvreader['event_date'] >= period_start.strftime("%Y-%m-%d")) & (csvreader['event_date'] < period_end.strftime("%Y-%m-%d"))]
            for key in monthly_df['admin2'].unique():
                    md_df = monthly_df[monthly_df['admin2'] == key]
                    countbattle = len(md_df[md_df['event_type'] == 'Battles'])
                    strikegov = len(md_df[(md_df['event_type'] == 'Explosions/Remote violence') & (md_df['inter1']==1)])
                    strikerebel = len(md_df[(md_df['event_type'] == 'Explosions/Remote violence') & (md_df['inter1']==2)])
                    civgov = len(md_df[(md_df['event_type'] == 'Violence against civilians') & (md_df['inter1']==1)])
                    civrebel = len(md_df[(md_df['event_type'] == 'Violence against civilians') & (md_df['inter1']==2)])

                    zx = recursion(admin2_dict[key]['control'],[strikerebel,civgov,countbattle,civrebel,strikegov]) #Reb. explosion,	Gov. Violence,	Armed clash,	Rebel violence,	Gov. explosion
                    admin2_dict[key]['control'] = zx

        m = folium.Map(location=[28.0, 29.0], zoom_start=6)

        
        
        for key, value in admin2_dict.items():
            # Extract color and other necessary values from the dictionary
            color = color_palette[value['control']]
            latitude = value['latitude']
            longitude = value['longitude']
            with open("correlation.txt", "a") as myfile:
                myfile.write(key + ' ' + str(colors_dict[value['control']]+1) + ' ' + str(read_color(latitude,longitude))+'\n')
            
            # Check if color is equal to 0
            if color == 0:
                color = (255, 255, 255)
            (r, g, b) = color
            
            # Create a folium Popup
            popup = folium.Popup(key, parse_html=True)
            
            # Create a CircleMarker and add it to the map
            if (r, g, b) != (255, 255, 255):
                folium.CircleMarker(location=[latitude, longitude],
                                    radius=5,
                                    color='#{:02x}{:02x}{:02x}'.format(0, 0, 0),
                                    fill=True,
                                    fill_opacity=1,
                                    fill_color='#{:02x}{:02x}{:02x}'.format(r, g, b),
                                    popup=popup).add_to(m)

        
        
        m.save("afg_full.html")

#myfunction_csv("pakistan.csv")

def read_correlation_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                data.append((parts[0], int(parts[1]), int(parts[2])))
    return data

def count_occurrences(data):
    occurrences = [[0, 0, 0] for _ in range(5)]  # Initialize a 5x3 table with zeros
    for _, num1, num2 in data:
        occurrences[num1 - 1][num2 // 2 - 1] += 1  # Increment corresponding cell in the table
    return occurrences

def print_table(occurrences):
    print("  1  3  5")
    for i, row in enumerate(occurrences, start=1):
        print(i, end=" ")
        for num in row:
            print(num, end=" ")
        print()

def calculate_correlation_ratio(data):
    total_rows = len(data)
    sum_x = sum(x for _, x, _ in data)
    sum_y = sum(y for _, _, y in data)
    sum_xy = sum(x * y for _, x, y in data)
    sum_x_squared = sum(x ** 2 for _, x, _ in data)
    sum_y_squared = sum(y ** 2 for _, _, y in data)

    numerator = total_rows * sum_xy - sum_x * sum_y
    denominator = ((total_rows * sum_x_squared - sum_x ** 2) * (total_rows * sum_y_squared - sum_y ** 2)) ** 0.5

    if denominator == 0:
        return float('inf')
    else:
        return numerator / denominator

# Main code
def counting_correlation():
    file_path = "correlation.txt"
    data = read_correlation_file(file_path)
    occurrences = count_occurrences(data)
    print("Table of occurrences:")
    print_table(occurrences)
    correlation_ratio = calculate_correlation_ratio(data)
    print("\nCorrelation ratio:", correlation_ratio)

#counting_correlation()


def plot():
    # Read the first file
    df_colors = pd.read_csv("result_with_colors.txt", sep=": ", header=None, names=["District", "Color"])
    df_colors['District'] = df_colors['District'].apply(lambda x: x.split(',')[0])

    # Read the second file
    with open("your_file.txt", "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.strip().split()
        district = ' '.join(parts[:-5])
        values = list(map(int, parts[-5:]))
        data.append([district] + values)

    df_values = pd.DataFrame(data, columns=["District", "Value_1", "Value_2", "Value_3", "Value_4", "Value_5"])

    # Merge the two dataframes on the 'District' column
    df_merged = pd.merge(df_colors, df_values, on="District")
    df_merged = df_merged[df_merged['Color']!='grey']

    df_merged["new color"] = df_merged.apply(lambda x: recursion(x.Color, [x.Value_1,x.Value_2,x.Value_3,x.Value_4,x.Value_5]), axis=1, result_type='expand')




    # Group by 'Color' and sum the values for each group
    #df_group = df_merged.groupby("Color").size().reset_index(name='Count2')
    #df_grouped = df_merged.groupby("Color").sum()
    #df_grouped = pd.merge(df_group, df_grouped, on=["Color"])
    #zero_value_filter = (df_merged['Value_1'] == 0) & \
    #                (df_merged['Value_2'] == 0) & \
    #                (df_merged['Value_3'] == 0) & \
    #                (df_merged['Value_4'] == 0) & \
    #                (df_merged['Value_5'] == 0)
    #df_zero = df_merged[zero_value_filter].copy()
    #df_zero = df_zero.groupby('Color').size().reset_index(name='Count')
    #df_grouped = pd.merge(df_grouped, df_zero, on=["Color"], how='outer')


    #df_grouped['rowszero'] = (df_merged.groupby('Color')[["Value_1", "Value_2", "Value_3", "Value_4", "Value_5"]].sum() == 0).sum(axis=1)

    df_merged.to_csv("new column.csv")


def foliumf():
    marker_cluster = folium.Map(location=[35.0, 66.0], zoom_start=8)


    # Read the content of the txt file and process each row
    with open("result_with_colors.txt", "r") as file:
        lines = file.readlines()

    # Add markers for each location to the map
    for line in lines:
        # Split the line into location and color
        location, color = line.strip().split(": ")
        latitude, longitude = map(float, location.split(", ")[1:])
        
        # Create a marker with the given color and label
        


        folium.CircleMarker(
            location=[latitude, longitude],
            radius=5,
            color=color_palette[color],
            fill=True,
            fill_color='#{:02x}{:02x}{:02x}'.format(color_palette[color][0],color_palette[color][1],color_palette[color][2]),
            fill_opacity=1,
            popup=location.split(",")[0],  # Use the first word of location as label
        ).add_to(marker_cluster)

    # Save the map as an HTML file
    marker_cluster.save("afghanistan_map.html")
#foliumf()


#df = pd.read_csv('pakistan.csv')

# Filter rows where "conflict_new_id" column is equal to 14129
#filtered_df = df[df['event_date'] < '2017-05-31']
#filtered_df = filtered_df[filtered_df['event_date'] > '2017-03-01']

# Write the filtered data to a new CSV file
#filtered_df.to_csv('afghanistan2.csv', index=False)



def funny():
    image = Image.open("Af2.png")
    csvreader = pd.read_csv("pakistan.csv")
    with open("result2.txt", "w") as f:
        for admin2 in csvreader["admin2"].unique():
            row = csvreader[(csvreader['admin2'] == admin2) & (csvreader['geo_precision'] == 2)].iloc[0]
            latitude = row["latitude"]
            longitude = row["longitude"]

            pixel_coords = coordtopx(latitude, longitude)
            avg_color = get_average_color(image, *pixel_coords, radius=1)
            f.write(f"{admin2}, {latitude}, {longitude}: {avg_color}\n")




def funny2():
    image = Image.open("Af.png")
    csvreader = pd.read_csv("pakistan.csv")
    with open("result.txt", "w") as f:
        for admin2 in csvreader["admin2"].unique():
            row = csvreader[csvreader["admin2"] == admin2].iloc[0]
            latitude = row["latitude"]
            longitude = row["longitude"]

            pixel_coords = coordtopx(latitude, longitude)
            avg_color = get_average_color(image, *pixel_coords, radius=1)
            color = ast.literal_eval(avg_color)
            
            # Find the closest predefined color
            color_palette2 = {
    1: (185,59,92),
    3: (246,160,142),
    5: (144,151,160)
}
            closest_color = find_closest_color(color, color_palette2)
            return closest_color

            #34.0926,64.0934   37.248,70.9906

#funny()
#do()

#file_path = "car.csv"
#csvreader = pd.read_csv(file_path, usecols=['event_date','event_type','sub_event_type','actor1','actor2','inter1','inter2','admin1','admin2','admin3','location','latitude','longitude','geo_precision'])

#file_path = "terrorism.xlsx"


# Read the data from data.xlsx
#df = pd.read_excel("terrorism.xlsx")

# Filter rows where 'provstate' is equal to 'North-West' or 'South-West'
#filtered_df = df[df['provstate'].isin(['North-West', 'South-West'])]

# Export the filtered data to new.xlsx
#filtered_df.to_csv("ambazonia2.csv", index=False, mode='a', header=False)



def replace_values(x):
        if x == 1 or x == 3555 or x == 3556:
            return 7
        elif x== 86:
            return 1
        else:
            return 2

def find_district2(village_name, province): #Bamenda I
    geolocator = Nominatim(user_agent="abcd")
    location = geolocator.geocode(village_name + ", " + province + ", " + "Cameroon")
    if location:
        xz = str(location).split(', ')
        if len(xz) == 4:
            return xz[0]
        else:
            return xz[1]
    else:
        return "Location not found."
    
def find_district1(village_name, province): #Mezam
    geolocator = Nominatim(user_agent="abcd")
    location = geolocator.geocode(village_name + ", " + province + ", " + "Cameroon")
    if location:
        xz = str(location).split(', ')
        if len(xz) == 4:
            return xz[1]
        else:
            return xz[-3]

        return str(location).split(', ')[1]
        #else:
        #    return "District information not found."
    else:
        return "Location not found."

def afghanistan():
    csvreader = pd.read_csv("afghanistan2.csv")
    csvreader2 = pd.read_csv("pakistan.csv")
    admins = csvreader2['admin2'].unique()

    #admin2_dict = {value: {'battle': 0, 'strikegov':0,'strikerebel':0,'civgov':0,'civrebel':0} for value in admins}


    for admin in admins:
        district_df = csvreader[csvreader['admin2'] == admin]
        countbattle = len(district_df[district_df['event_type'] == 'Battles'])
        strikegov = len(district_df[(district_df['event_type'] == 'Explosions/Remote violence') & (district_df['inter1']==1)])
        strikerebel = len(district_df[(district_df['event_type'] == 'Explosions/Remote violence') & (district_df['inter1']==2)])
        civgov = len(district_df[(district_df['event_type'] == 'Violence against civilians') & (district_df['inter1']==1)])
        civrebel = len(district_df[(district_df['event_type'] == 'Violence against civilians') & (district_df['inter1']==2)])

        with open("your_file.txt", "a") as file:
            file.write(f"{admin} {countbattle} {strikegov} {strikerebel} {civgov} {civrebel}\n")


    #csvreader.to_csv("afghanistan.csv", index=False)
#afghanistan()

def gtd():
    df = pd.read_csv("ambazonia2.csv",usecols=['iyear','imonth','iday','provstate','city','latitude','longitude','specificity','summary','targtype1_txt','gname','attacktype1_txt'],index_col=False)

    df = df[df['gname'] != 'Unknown']
    df = df[df['specificity'] < 4]

    df['event_date'] = df[['iyear', 'imonth', 'iday']].astype(str).agg('-'.join, axis=1)

    # Convert the date format to "YYYY-MM-DD"
    #df['event_date'] = df['event_date'].dt.strftime('%Y-%m-%d')

    # Drop the original columns if needed
    df = df.drop(['iyear', 'imonth', 'iday'], axis=1)

    df.rename(columns={'summary': 'notes'}, inplace=True)
    df.rename(columns={'gname': 'actor1'}, inplace=True)
    df.rename(columns={'provstate': 'admin1'}, inplace=True)
    df.rename(columns={'targtype1_txt': 'actor2'}, inplace=True)


    def newfun(arg1,arg2):
        xz = 0
        if arg1 == 'Private Citizens & Property' or arg1 == 'Educational Institution' or arg1 == 'Religious Figures/Institutions' or arg1 == 'NGO':
            xz = 7
        if arg1 in ['Police','Military','Government (General)']:
            xz = 1
        if arg2 == 'Bombing/Explosion':
            return ['Explosions/Remote violence','Remote explosive/landmine/IED',xz]
        if xz==7:
            return ['Violence against civilians','Attack',xz]
        if xz==1:
            return ['Battles','Armed clash',xz]
        return ['','',0]
        
    df = df.assign(event_type=0)
    df = df.assign(sub_event_type=0)
    df = df.assign(inter2=0)    
    #df.index = [x for x in range(1, len(df.values)+1)]

    df[['event_type','sub_event_type','inter2']] = df.apply(lambda x: newfun(x.actor2, x.attacktype1_txt), axis=1, result_type='expand')   #    df[['actor2','attacktype1_txt']].apply(newfun, axis=1)
    #zx = find_district()
    def find_district1(village_name):
        geolocator = Nominatim(user_agent="abcd")
        location = geolocator.geocode(village_name + ", " + "Cameroon")
        if location:
            return str(location).split(', ')[2]
            #else:
            #    return "District information not found."
        else:
            return "Location not found."
    
    df['admin2'] = df['city'].apply(find_district1)
    df['admin3'] = df['city'].apply(find_district2)
    df = df.drop(['attacktype1_txt'], axis=1)
    df.rename(columns={'city': 'location'}, inplace=True)
    df.rename(columns={'specificity': 'geo_precision'}, inplace=True)
    df = df.assign(inter1=2)
    df.to_csv('ambazoniaterror.csv', index=False)  

def filtergd():
    # Read the original CSV file
    df = pd.read_csv('gdset.csv')

    # Filter rows where "conflict_new_id" column is equal to 14129
    filtered_df = df[df['conflict_new_id'] == 14129]

    # Write the filtered data to a new CSV file
    filtered_df.to_csv('gdsetambazonia.csv', index=False)



def remove_last_word(input_string):
    words = input_string.split()
    if len(words) > 1:
        return ' '.join(words[:-1])
    else:
        # If there's only one word, return an empty string
        return ''



def uppsala2():
    df = pd.read_csv("outambazonia.csv")
    df['admin3'] = df.apply(lambda x: find_district2(x.location, x.admin2),axis=1)
    df.to_csv('outambazonia.csv', index=False) 


def terror2():
    df = pd.read_csv("ambazoniaterror.csv")
    #df['admin2'] = df.apply(lambda x: find_district1(x.location, x.admin1),axis=1)
    df['admin3'] = df.apply(lambda x: find_district2(x.location, x.admin1),axis=1)
    df.to_csv('ambazoniaterror.csv', index=False) 

#terror2()
#uppsala2()

def uppsala():
    df = pd.read_csv("gdsetambazonia.csv",usecols=['type_of_violence','side_a',"side_a_new_id","side_b","side_b_new_id",'where_prec','where_coordinates','adm_1','adm_2','latitude','longitude','date_end'],index_col=False)


    df['date_end'] = df['date_end'].str.split(' ').str[0]



    def replace_values(x):
        if x == 1:
            return 7
        elif x== 83:
            return 1
        else:
            return 2
        

    def replace_values2(x):
        if x == 1 or x==2:
            return "Armed clash"
        elif x==3:
            return "Violence against civilians"
        
    df['side_a_new_id'] = df['side_a_new_id'].apply(replace_values)
    df['side_b_new_id'] = df['side_b_new_id'].apply(replace_values)

    df['type_of_violence'] = df['type_of_violence'].apply(replace_values2)

    df = df[df['where_prec'] < 5]
    df['where_prec'] = df['where_prec'].replace(2, 1)
    df['where_prec'] = df['where_prec'].replace(3, 2)
    df['where_prec'] = df['where_prec'].replace(4, 3)

    df = df.dropna(subset=['adm_2'])


    df['adm_1'] = df['adm_1'].str.split(' ').str[0]
    df['adm_2'] = df['adm_2'].str.split(' ').str[0]
    df['where_coordinates'] = df['where_coordinates'].apply(remove_last_word)
    df['admin3'] = df.apply(lambda x: find_district2(x.where_coordinates, x.adm_2))   #  df['where_coordinates'].apply(find_district2)

    df.rename(columns={'side_a': 'actor1'}, inplace=True)
    df.rename(columns={'side_b': 'actor2'}, inplace=True)
    df.rename(columns={'side_a_new_id': 'inter1'}, inplace=True)
    df.rename(columns={'side_b_new_id': 'inter2'}, inplace=True)
    df.rename(columns={'date_end': 'event_date'}, inplace=True)
    df.rename(columns={'where_prec': 'geo_precision'}, inplace=True)
    df.rename(columns={'where_coordinates': 'location'}, inplace=True)
    df.rename(columns={'adm_1': 'admin1'}, inplace=True)
    df.rename(columns={'adm_2': 'admin2'}, inplace=True)



    df.rename(columns={'type_of_violence': 'sub_event_type'}, inplace=True)


    df.to_csv('outambazonia.csv', index=False)  

#uppsala()