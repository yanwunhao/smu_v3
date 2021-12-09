import numpy as np
import matplotlib.pyplot as plt

import csv
import os

with open('./raw_data/smu_data.csv', 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    rows = [row for row in csv_reader]
    samples = rows
    # samples = [[row[col] for row in rows] for col in range(len(rows[0]))]
    f.close()

samples = np.array(samples)

with open('./raw_data/offset_data.csv', 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    rows = [row for row in csv_reader]
    offsets = rows
    # offsets = [[row[col] for row in rows] for col in range(len(rows[0]))]
    f.close()

offsets = np.array(offsets)

selected_samples_numbers = offsets.T[0]

selected_samples = []

handled_sample_list = []
handled_curve = []

for sample in samples:
    sample_number = sample[0]
    if sample_number in selected_samples_numbers:
        selected_samples.append(sample)
    else:
        print(sample_number + ' offset is missing')

for i in range(len(selected_samples)):
    sample = selected_samples[i]
    offset = offsets[i]
    if sample[0] != offset[0]:
        print(sample[0] + ' number is not corresponding to offset number')
        break
    else:
        number = sample[0]
        bci = float(sample[1])
        booi = float(sample[2])
        disease_type = sample[3]
        vv = float(sample[4])
        qmax = float(sample[5])
        qavg = float(sample[6])
        pvr = int(sample[7])
        vt = float(sample[8])
        ft = float(sample[9])

        offset_value = float(offset[1])
        device = int(offset[2])

        filename = './raw_data/csv_files/' + number + '.csv'

        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8-sig') as f:
                csv_reader = csv.reader(f)
                rows = [row for row in csv_reader]
                curve_data = np.array(rows, dtype=float).T
            f.close()

            x_series = curve_data[0]
            y_series = curve_data[1]

            if len(x_series) != len(y_series):
                print(number + ' 2-dimensional array not correspond')
                break

            # Fix the curve with offset
            x_series = x_series - offset_value
            indexes_to_remove = np.where(x_series < 0)
            x_series = np.delete(x_series, indexes_to_remove)
            y_series = np.delete(y_series, indexes_to_remove)

            # perform a definite integral on curve among [0,vt]
            start = x_series[0]
            end = x_series[-1]

            resampling_x = np.linspace(start, end, 200)
            resampling_y = np.interp(resampling_x, x_series, y_series)

            x_series = resampling_x
            y_series = resampling_y

            for y in range(len(y_series)):
                if y_series[y] < 0:
                    y_series[y] = 0

            # draw figs
            plot_x = range(200)
            plt.plot(plot_x, y_series, color='black', linewidth=2)
            plt.savefig('./figs/' + number + '.jpg')

            qmax_grabbed = np.amax(y_series)

            delta_t = vt / 200.
            vv_integral = 0

            for j in range(len(y_series)):
                vv_integral += delta_t * y_series[j]

            if device != 3:
                vt = vt - offset_value

            qavg_by_vt = vv/vt
            qavg_by_ft = vv/ft

            qavg_by_vt_integral = vv_integral/vt
            qavg_by_ft_integral = vv_integral/ft

            handled_parameter = [number, bci, booi,
                                 disease_type, vv, vv_integral,
                                 qmax, qmax_grabbed,
                                 qavg, qavg_by_vt, qavg_by_ft,
                                 qavg_by_vt_integral, qavg_by_ft_integral,
                                 pvr, vt, ft]
            handled_sample_list.append(handled_parameter)

            y_series = np.append(number, y_series)
            handled_curve.append(y_series)
        else:
            print(number + ' curve data is missing')

with open('./data/handled_parameter.csv', 'w') as f:
    csv_writer = csv.writer(f)
    for sample in handled_sample_list:
        csv_writer.writerow(sample)
    f.close()

with open('./data/handle_curve.csv', 'w') as f:
    csv_writer = csv.writer(f)
    for curve in handled_curve:
        csv_writer.writerow(curve)
    f.close()
