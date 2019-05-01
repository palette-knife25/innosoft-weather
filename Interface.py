"""
This code provides interface for weather prediction
It accepts current parameters as strings (model_path, time) and numbers (all others)
It returns JSON prediction

Author: Dinar Salakhutdinov
"""
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import sys


def predict_conditions(model_path, time, wind_speed, wind_gusts, dew_point_temperature, road_temperature,
                       underground_temperature, air_temperature, snow_intensity, rain_intensity, humidity, hoursdelta=3):
    # Convert arguments to dataframe needed for model:
    columns = ['wind_speed', 'wind_gusts', 'dew_point_temperature', 'road_temperature', 'underground_temperature',
               'air_temperature',
               'snow_intensity', 'rain_intensity', 'humidity']
    x = np.array(columns)
    row = [wind_speed, wind_gusts, dew_point_temperature, road_temperature, underground_temperature, air_temperature,
           snow_intensity,
           rain_intensity, humidity]
    x = np.vstack([x, row])
    dataframe = pd.DataFrame(columns=x[0], data=x[1:])

    # Add three hours to initial time:
    time_end = datetime.strptime(time, "%Y-%m-%d %H:%M UTC")
    time_end = time_end + timedelta(hours=hoursdelta)

    # Load saved model:
    with open(model_path, 'rb') as h:
        loaded_model = pickle.load(h)

    if loaded_model is None:
        print("Error in Interface: Model file is not found")
        sys.exit()

    # Predict values:
    predicted = loaded_model.predict(dataframe)

    # Create dictionary in roadcast format:
    roadcast_dict = dict()
    roadcast_dict['timestamp'] = str(time_end)
    roadcast_dict['wind_speed'] = predicted[0][0].item()
    roadcast_dict['wind_gusts'] = predicted[0][1].item()
    roadcast_dict['dew_point_temperature'] = predicted[0][2].item()
    roadcast_dict['road_temperature'] = predicted[0][3].item()
    roadcast_dict['underground_temperature'] = predicted[0][4].item()
    roadcast_dict['air_temperature'] = predicted[0][5].item()
    roadcast_dict['snow_intensity'] = predicted[0][6].item()
    roadcast_dict['rain_intensity'] = predicted[0][7].item()
    roadcast_dict['humidity'] = predicted[0][8].item()

    return json.dumps(roadcast_dict)
