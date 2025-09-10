import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import os.path as osp
import pandas as pd
import numpy as np
from scipy import interpolate

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



qm9s_dataset_info = {
    'name': 'qm9s',
    'max_num_atoms_peak_H': 18, 'max_num_atoms_peak_C': 9,
    'min_peak_H': -5, 'max_peak_H': 20,
    'min_peak_C': -10, 'max_peak_C': 290,
    
}



def prep_ir_geo_dataset(data_dir, dataset, split_file='', test_run=False):
    
   
    dataset_dic = {"train": None, "val":None, "test":None}

    if split_file != '':
        df_all = pd.read_pickle(osp.join(data_dir, split_file))
        df_all = df_all.sample(frac=1)
        print(df_all)
        
        test_df = df_all.iloc[:1000]
        dataset_dic["test"] = NMRDataset(test_df, dataset=dataset)
        test_df.to_pickle(osp.join(data_dir, "test_1000_qm9s.pkl"))
        
        train_val_df = df_all.iloc[1000:]
        num_train = int(0.95*len(train_val_df))
        train_df = train_val_df.iloc[:num_train]
        dataset_dic["train"] = NMRDataset(train_df, dataset=dataset)
        train_df.to_pickle(osp.join(data_dir, f"train_{num_train}_qm9s.pkl"))
        
        val_df = train_val_df.iloc[num_train:]
        dataset_dic["val"] = NMRDataset(val_df, dataset=dataset)
        val_df.to_pickle(osp.join(data_dir, f"val_{len(val_df)}_qm9s.pkl"))


    else:
        print("No new datasets are splitted.")
        files = os.listdir(data_dir)
        if test_run: 
            for file_name in files:
                if "val_" in file_name: break
            return NMRDataset(pd.read_pickle(osp.join(data_dir,file_name)))

        for file_name in files:
            if "train_" in file_name and '.pkl' in file_name:
                train_df = pd.read_pickle(osp.join(data_dir,file_name))
                dataset_dic["train"] = NMRDataset(train_df, dataset=dataset)
            elif "val_" in file_name and '.pkl' in file_name:
                val_df = pd.read_pickle(osp.join(data_dir,file_name))
                dataset_dic["val"] = NMRDataset(val_df, dataset=dataset)
            elif "test_" in file_name and '.pkl' in file_name:
                test_df = pd.read_pickle(osp.join(data_dir,file_name))
                dataset_dic["test"] = NMRDataset(test_df, dataset=dataset)
    
    return dataset_dic



def process_spec(origin_y, origin_x):
    new_x = np.linspace(500, 4000, 3200)
    intp_func = interpolate.interp1d(origin_x, origin_y)
    new_y = intp_func(new_x)

    # normalization of ir spectra
    new_y = new_y / np.max(new_y) * 99
        
    new_y_int = new_y.astype(int)
    new_y_int = np.clip(new_y_int, 0, 99)

    return new_y_int


class NMRDataset(Dataset):
    def __init__(self, data_df, dataset='qm9s'):
        """
            data_df: [id, charges, smiles, y_H, x_grouped_H, y_grouped_H, num_peaks_H, y_C, x_grouped_C, y_grouped_C, num_peaks_C]
            dataset: qm9s
            mode: H, C

        numpy array to torchtensor
        """
        self.data_df = data_df
       

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        data_i = self.data_df.iloc[idx]

        data_i_ = {}
        for key, value in data_i.to_dict().items():
            if key in ["id", "smiles"]: 
                data_i_[key] = value
            else:
                data_i_[key] = torch.from_numpy(value)
        
        return data_i_


def dataset_collate_fn(data, dataset):
    """
    collate_fn receives a list of tuples if your __getitem__ function from a Dataset.
    data:
        a list of dictionary.
    """
    num_peaks_H = torch.stack([data_i["num_peaks_H"] for data_i in data])
    max_num_peaks_H = max(num_peaks_H)
    num_peaks_C = torch.stack([data_i["num_peaks_C"] for data_i in data])
    max_num_peaks_C = max(num_peaks_C)
    
    batch_size = num_peaks_H.size(0)
    
    if dataset == 'qm9s':
        dataset_info = qm9s_dataset_info

    peak_H_one_hot = torch.zeros(batch_size, max_num_peaks_H, dataset_info[dataset]['max_num_atoms_peak_H'])
    peak_C_one_hot = torch.zeros(batch_size, max_num_peaks_C, dataset_info[dataset]['max_num_atoms_peak_C'])

    x_grouped_H = torch.zeros(batch_size, max_num_peaks_H)
    x_grouped_C = torch.zeros(batch_size, max_num_peaks_C)
    
    for i in range(batch_size):
        peak_H_one_hot[i, :num_peaks_H[i], :] = F.one_hot(data[i]["y_grouped_H"], dataset_info[dataset]['max_num_atoms_peak_H'])
        peak_C_one_hot[i, :num_peaks_C[i], :] = F.one_hot(data[i]["y_grouped_C"], dataset_info[dataset]['max_num_atoms_peak_C'])
        x_grouped_H[i, :num_peaks_H[i]] = data[i]["x_grouped_H"]
        x_grouped_C[i, :num_peaks_C[i]] = data[i]["x_grouped_C"]
        
    nmr_h = torch.stack([data_i["y_H"] for data_i in data])
    nmr_c = torch.stack([data_i["y_C"] for data_i in data])
    

    

   

    return {
        "id": [data_i["id"] for data_i in data],
        "smiles": [data_i["smiles"] for data_i in data],
        "nmr_h": nmr_h,
        "nmr_c": nmr_c,
        "peak_H_one_hot": peak_H_one_hot,
        "peak_C_one_hot": peak_C_one_hot,
        "x_grouped_H": x_grouped_H,
        "x_grouped_C": x_grouped_C,
        }
    

def qm9s_dataset_collate_fn(data):
    return dataset_collate_fn(data, dataset='qm9s')








if __name__ == "__main__":
    dataset = prep_ir_geo_dataset()
    print(dataset[0])
    loader = DataLoader(dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset_collate_fn
            )
    for batch in loader:
        print(batch)
        break
    