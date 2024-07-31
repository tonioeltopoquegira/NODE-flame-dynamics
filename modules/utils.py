import pandas as pd
import os

folder_names = {
    ('050',) : 'Low_Amp',
    ('150',) : 'High_Amp',
    ('100',) : 'Med_Amp',
    ('050', '100', '150'): 'All_Amp',
    ('050', '100'): '100_50_Amp'
}

def write_summary(excel_filename, model_name, data_details, train_details, train_loss, test_loss, metrics_harmonics):
    run = {'model': model_name}
    run.update(data_details)
    run.update(train_details)
    run.update({
    'rMSE_train_loss': f'{train_loss:.3}', 
    'rMSE_test_loss': f'{test_loss:.3}'
    })
    run.update(metrics_harmonics)


    df = pd.DataFrame([run])

    if os.path.isfile(excel_filename):
        df_existing = pd.read_excel(excel_filename)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df

    summ = os.path.join("summaries", excel_filename)
    df_combined.to_excel(summ, index=False)

    print(f"Summary has been written to {excel_filename}!")

def set_up_folders(model_name, a):
    a = tuple(a)
    fname = folder_names[a]
    path = os.path.join(model_name, fname)
    directory_path = os.path.join("figures", path)
    os.makedirs(directory_path, exist_ok=True)
    directory_path = os.path.join(directory_path, "activations")
    os.makedirs(directory_path, exist_ok=True)
    directory_path = os.path.join("weights",path)
    os.makedirs(directory_path, exist_ok=True)
    return path
