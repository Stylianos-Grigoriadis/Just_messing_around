import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

# Warm_up_1
# Warm_up_2
# Warm_up_3
# Isometric_05_T1
# Isometric_05_T2
# Isometric_20_T1
# Isometric_20_T2
# Isometric_40_T1
# Isometric_40_T2
# Isometric_60_T1
# Isometric_60_T2
# Isometric_80_T1
# Isometric_80_T2
# Pert_down_T1
# Pert_down_T2
# Pert_up_T1
# Pert_up_T2


directory_path = r'C:\Users\Stylianos\Desktop\name_change'

ID = '5.Young'
excel_for_names = pd.read_excel(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip game\Pilot Study 10\Participants.xlsx')
index = excel_for_names[excel_for_names['ID'] == ID].index[0]
max_MVC = excel_for_names['MVC'][index]

def renaming_process(index,directory_path):

    Warm_up_order_list = ['Warm_up_1.csv', 'Warm_up_2.csv', 'Warm_up_3.csv']

    Iso_order = excel_for_names['Order Iso'][index]
    Iso_values = Iso_order.split(',')
    Iso_order_list = [f"Isometric_{value.zfill(2)}_T{i+1}.csv" for value in Iso_values for i in range(2)]

    Pert_order = excel_for_names['Order Pert'][index]
    Pert_values = Pert_order.split(',')
    counters = {'up': 0, 'down': 0}
    Pert_order_list = []
    for value in Pert_values:
        counters[value] += 1
        Pert_order_list.append(f"Pert_{value}_T{counters[value]}.csv")

    new_names_list = Warm_up_order_list + Iso_order_list + Pert_order_list

    files = glob.glob(os.path.join(directory_path, "*"))

    names = []
    for i in range(len(files)):
        old_file_name = os.path.basename(files[i])
        old_file_path = os.path.join(directory_path, old_file_name)
        new_file_path = os.path.join(directory_path, new_names_list[i])
        os.rename(old_file_path, new_file_path)


def showing_plots(max_MVC,directory_path):
    percentage_05_T1 = 0.05 * max_MVC
    percentage_05_T2 = 0.05 * max_MVC
    percentage_20_T1 = 0.2 * max_MVC
    percentage_20_T2 = 0.2 * max_MVC
    percentage_40_T1 = 0.4 * max_MVC
    percentage_40_T2 = 0.4 * max_MVC
    percentage_60_T1 = 0.6 * max_MVC
    percentage_60_T2 = 0.6 * max_MVC
    percentage_80_T1 = 0.8 * max_MVC
    percentage_80_T2 = 0.8 * max_MVC
    percentage_Pert_down_T1 = 0.3 * max_MVC
    percentage_Pert_down_T2 = 0.3 * max_MVC
    percentage_Pert_up_T1 = 0.3 * max_MVC
    percentage_Pert_up_T2 = 0.3 * max_MVC
    percentage_Warm_up_1 = 0.2 * max_MVC
    percentage_Warm_up_2 = 0.2 * max_MVC
    percentage_Warm_up_3 = 0.2 * max_MVC
    list_percentages = [percentage_05_T1, percentage_05_T2, percentage_20_T1, percentage_20_T2, percentage_40_T1, percentage_40_T2, percentage_60_T1, percentage_60_T2, percentage_80_T1, percentage_80_T2, percentage_Pert_down_T1, percentage_Pert_down_T2, percentage_Pert_up_T1, percentage_Pert_up_T2, percentage_Warm_up_1, percentage_Warm_up_2, percentage_Warm_up_3]
    print(f'5% : {0.05*max_MVC}')
    print(f'20% : {0.2*max_MVC}')
    print(f'40% : {0.4*max_MVC}')
    print(f'60% : {0.6*max_MVC}')
    print(f'80% : {0.8*max_MVC}')
    files = glob.glob(os.path.join(directory_path, "*"))
    for i, file in enumerate(files):
        file_name = os.path.basename(file)
        signal = pd.read_csv(file, skiprows=2)
        plt.plot(signal['Performance'])
        plt.axhline(y=list_percentages[i], color='red')
        plt.title(f'{file_name}, target = {list_percentages[i]}\naverage(200:600) = {round(np.mean(signal["Performance"][200:600]),3)}')
        plt.show()

# renaming_process(index, directory_path)
# showing_plots(max_MVC,directory_path)