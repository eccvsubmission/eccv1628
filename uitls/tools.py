import os
from datetime import datetime
import pandas as pd

def make_results_folder(base_path: str = "results"):
    try:
        os.mkdir(base_path)
    except FileExistsError:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f"{base_path}/{timestamp}/"

    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    return folder_path

def store_results(folder_path, epoch, result_dict):
    try:
        df = pd.read_csv(f"{folder_path}results.csv")
    except:
        columns =  ["epoch"] + list(result_dict.keys())
        df = pd.DataFrame(columns=columns)
    result_values = [epoch] + list(result_dict.values())
    
    df.loc[len(df)] = result_values
    df.to_csv(f"{folder_path}results.csv", index=False) 

    
def save_parameters_to_txt(file_path, **kwargs):
    with open(file_path, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

