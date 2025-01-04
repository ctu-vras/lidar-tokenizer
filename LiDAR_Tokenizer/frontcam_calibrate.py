import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
from scipy.spatial.transform import Rotation as R
import os

channel2sensor = {'Horizon_left':2,
                  'Horizon_right':3,
                  'Tele15_center':4,
                  'Avia_left':5,
                  'Avia_center':6,
                  'Avia_right':7,
                  'Avia_tilted':9,}

rotations = {2:[0,0,0],
             3:[0,0,0],
             4:[0,0,0],
             5:[0,0,0],
             6:[0,0,0],
             7:[0,0,0],
             9:[0,0,0],}

translations = {2:[0,0,0],
                3:[0,0,0],
                4:[0,0,0],
                5:[0,0,0],
                6:[0,0,0],
                7:[0,0,0],
                9:[0,0,0],}

sensors = ['Livox_ID1','Livox_ID2','Livox_ID3','Livox_ID4','Livox_ID5','Livox_ID6','Livox_ID7']



if __name__ == "__main__":

    with open("data/dfc_golf5_config_ver_0_2_CALIBRATED.json", 'r') as file:
        calibration = json.load(file)

    for sensor in calibration['component_info']:
        sensor_config = calibration['component_info'][sensor]
        if 'Sensor name' in sensor_config:
            sensor_name = sensor_config['Sensor name']
        else:
            continue
        if sensor_name in channel2sensor.keys():
            channel = channel2sensor[sensor_name]
            translations[channel] = [sensor_config['Mounting position']['pos_x_m'],
                                    sensor_config['Mounting position']['pos_y_m'],
                                    sensor_config['Mounting position']['pos_z_m']]
            rotations[channel] =    [sensor_config['Mounting position']['yaw_deg'],
                                    sensor_config['Mounting position']['pitch_deg'],
                                    sensor_config['Mounting position']['roll_deg']]

    channels = [2,3,4,5,6,7,9]

    data_folders = ['/data/npz_2024-09-06-07-42-41',
                    '/data/npz_2024-09-06-10-59-33',
                    '/data/npz_2024-09-06-11-42-34',
                    '/data/npz_2024-09-09-12-33-35',
                    '/data/npz_2024-09-13-07-06-04',
                    '/data/npz_2024-09-13-10-31-13',
                    '/data/npz_2024-09-13-12-19-26',
                    ]

    for data_root in data_folders:
        for f in os.listdir(data_root):
            path = os.path.join(data_root, f)
            data = np.load(path)

            data_points = data['point_cloud'][:,:3]
            data_channels = data['point_cloud'][:,4]

            data_calibrated = np.zeros_like(data_points)

            for i in range(len(channels)):
                channel = channels[i]

                mask = data_channels == channel
                data_points_masked = data_points[mask,:]

                # Revert incorrect calibration
                if channel == 4:    # Tele15_center
                    rot = R.from_euler('x', 180, degrees=True).as_matrix()
                    data_points_masked = (rot @ data_points_masked.T).T

                if channel == 5:    # Avia_left
                    rot = R.from_euler('x', -90, degrees=True).as_matrix()
                    data_points_masked = (rot @ data_points_masked.T).T

                if channel == 6:    # Avia_center
                    rot = R.from_euler('x', -90, degrees=True).as_matrix()
                    data_points_masked = (rot @ data_points_masked.T).T

                if channel == 7:    # Avia_right
                    rot = R.from_euler('x', -90, degrees=True).as_matrix()
                    data_points_masked = (rot @ data_points_masked.T).T

                if channel == 9:    # Avia_tilted
                    rot = R.from_euler('xy', [180,13], degrees=True).as_matrix()
                    data_points_masked = (rot @ data_points_masked.T).T

                # Actual correct calibration.
                rot = R.from_euler('zyx', rotations[channel], degrees=True).as_matrix()
                data_points_masked = (rot @ data_points_masked.T).T + translations[channel]

                data_calibrated[mask,:3] = data_points_masked

            data_calibrated = np.hstack((data_calibrated,data['point_cloud'][:,3:]))
            data_calibrated = data_calibrated.astype(np.float32)
            np.savez(path, point_cloud=data_calibrated, timestamp=data['timestamp'])
