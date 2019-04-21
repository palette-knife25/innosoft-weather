import Interface
import json

predicted = Interface.predict_conditions(model_path='models/extra_trees_model.sav', time='2019-02-13 09:01 UTC',
                                         wind_speed=3.3,
                                         wind_gusts=4, dew_point_temperature=-13, road_temperature=-11.6,
                                         underground_temperature=-13.3, air_temperature=-8.4, snow_intensity=0.0,
                                         rain_intensity=0.0, humidity=69.3)

print(predicted, '\n')

data = json.loads(predicted)
print('Road temperature:', data['road_temperature'])
print('Air temperature:', data['air_temperature'])
print('Rain intensity:', data['rain_intensity'])
print('Snow intensity:', data['snow_intensity'])
