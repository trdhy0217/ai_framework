import os
import time
import argparse
import numpy as np
from scipy.io import savemat, loadmat

size_field = 128
x_range = [-0.2, 0.7]
y_range = [-0.3, 0.3]
interval_xy = round(((x_range[1] - x_range[0]) / (size_field - 1)), 4)

def plot_flow_field(prediction_dir, flow_type, data_id):
    flow_field_class = ['real', 'prediction']
    flow_field_type = ['P', 'U', 'V']
    flow_field_data = loadmat(f'{prediction_dir}/prediction_data{data_id}.mat')

    flow_dir = os.path.join(prediction_dir, flow_field_type[flow_type])
    if not os.path.exists(flow_dir):
        os.mkdir(flow_dir)

    for c in flow_field_class:
        flow_field = flow_field_data[f"{c}"]
        for n in range(len(flow_field)):
            with open(f"{flow_dir}/{flow_field_type[flow_type]}_{c}_{n + 1}.dat", "w+") as file:
                file.write(f"VARIABLES = X, Y,{flow_field_type[flow_type]}_{c}\n")
                file.write(f"zone i={size_field} j={size_field}\n")
                single_flow = np.squeeze(flow_field[n, flow_type, :, :])
                for i in range(size_field):
                    x = x_range[0] + i * interval_xy
                    for j in range(size_field):
                        y = y_range[0] + j * interval_xy
                        file.write(f"{x} {y} {single_flow[i, j]}\n")
                file.close()
                
                
prediction_dir = './save_prediction_continue'
data_id = 4
for i in range(3):
    plot_flow_field(prediction_dir, i, data_id)