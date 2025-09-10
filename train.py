import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from dataset_dataloader import prep_ir_geo_dataset, qm9s_dataset_collate_fn
from nmr_cnn_encoder import NMR_CNN_Encoder_PeakPrediction

def main():
    dataset = prep_ir_geo_dataset(dataset='qm9s')
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=qm9s_dataset_collate_fn)
    model = NMR_CNN_Encoder_PeakPrediction()
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, loader)