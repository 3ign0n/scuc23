import os
import pandas as pd
from datetime import datetime

def main():
    targets = {
        '42':'2023-08-20T14.32.36.301Z',
        '13':'2023-08-20T15.01.32.351Z',
        '57':'2023-08-20T15.05.30.002Z',
        '80':'2023-08-20T15.07.43.229Z',
        '96':'2023-08-20T15.14.03.201Z',
        '0':'2023-08-20T22.39.19.494Z',
        '7':'2023-08-20T22.42.18.024Z',
        '101':'2023-08-20T22.44.44.432Z',
        '108':'2023-08-20T22.46.52.955Z',
        '119':'2023-08-20T22.48.50.264Z',
        }
    base_dir = 'data/07_model_output/y_pred.csv'

    csv_files = [os.path.join(base_dir, target, 'y_pred.csv') for target in targets.values()]
    df_list = [pd.read_csv(file, header=None) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True, axis=1)
    df = df.drop(df.columns[[2, 4, 6, 8, 10, 12, 14, 16, 18]],axis = 1)
    df['mean'] = df[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]].mean(axis=1)
    df = df.drop(df.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],axis = 1)

    start_datetime = datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")[:-3] + 'Z'
    output_dir=os.path.join(base_dir, start_datetime)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'y_pred.csv'), header=None, index=False)

if __name__ == '__main__':
    main()
