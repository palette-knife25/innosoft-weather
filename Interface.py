import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json


def predict_conditions(model_path, time, wind_speed, wind_gusts, dew_point_temperature, road_temperature,
                       underground_temperature, air_temperature, snow_intensity, rain_intensity, humidity):
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
    time_end = time_end + timedelta(hours=3)

    # Load saved model:
    with open(model_path, 'rb') as h:
        loaded_model = pickle.load(h)

    # Predict values:
    predicted = loaded_model.predict(dataframe)

    # Create dictionary in roadcast format:
    roadcast_dict = dict()
    roadcast_dict['timestamp'] = str(time_end)
    roadcast_dict['wind_speed'] = predicted[0][0]
    roadcast_dict['wind_gusts'] = predicted[0][1]
    roadcast_dict['dew_point_temperature'] = predicted[0][2]
    roadcast_dict['road_temperature'] = predicted[0][3]
    roadcast_dict['underground_temperature'] = predicted[0][4]
    roadcast_dict['air_temperature'] = predicted[0][5]
    roadcast_dict['snow_intensity'] = predicted[0][6]
    roadcast_dict['rain_intensity'] = predicted[0][7]
    roadcast_dict['humidity'] = predicted[0][8]

    return json.dumps(roadcast_dict)
