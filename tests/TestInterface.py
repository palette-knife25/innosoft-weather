import Interface
import unittest
import json
from datetime import datetime, timedelta


class TestInterface(unittest.TestCase):

    def test_interface(self):

        # Test parameters:
        model_path = '../models/extra_trees_model.sav'
        time = '2019-02-13 09:01 UTC'
        wind_speed = 3.3
        wind_gusts = 4
        dew_point_temperature = -13
        road_temperature = -11.6
        underground_temperature = -13.3
        air_temperature = -8.4
        snow_intensity = 0.0
        rain_intensity = 0.0
        humidity = 69.3

        predicted = Interface.predict_conditions(model_path=model_path, time=time,
                                                 wind_speed=wind_speed,
                                                 wind_gusts=wind_gusts, dew_point_temperature=dew_point_temperature,
                                                 road_temperature=road_temperature,
                                                 underground_temperature=underground_temperature,
                                                 air_temperature=air_temperature,
                                                 snow_intensity=snow_intensity,
                                                 rain_intensity=rain_intensity, humidity=humidity)

        data = json.loads(predicted)

        self.assertIsNotNone(predicted), "Interface returned no prediction"

        self.assertIsNotNone(data['timestamp'])
        time_end = datetime.strptime(time, "%Y-%m-%d %H:%M UTC") + timedelta(hours=3)
        self.assertEqual(str(time_end), data['timestamp']), "Prediction does not contain timestamp"

        self.assertIsNotNone(data['wind_speed']), "Prediction does not contain wind_speed"
        self.assertIsNotNone(data['wind_gusts']), "Prediction does not contain wind_gusts"
        self.assertIsNotNone(data['dew_point_temperature']), "Prediction does not contain dew_point_temperature"
        self.assertIsNotNone(data['road_temperature']), "Prediction does not contain road_temperature"
        self.assertIsNotNone(data['underground_temperature']), "Prediction does not contain underground_temperature"
        self.assertIsNotNone(data['air_temperature']), "Prediction does not contain air_temperature"
        self.assertIsNotNone(data['snow_intensity']), "Prediction does not contain snow_intensity"
        self.assertIsNotNone(data['rain_intensity']), "Prediction does not rain_intensity"
        self.assertIsNotNone(data['humidity']), "Prediction does not contain humidity"
