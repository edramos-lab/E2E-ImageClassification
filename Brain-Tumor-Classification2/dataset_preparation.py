from unittest import skip
import cv2
import numpy
import os
import click
import pandas as pd



@click.command()
@click.option('--dataset_dir', prompt='Enters the dataset parent directory')



def dataset_preparation(dataset_dir):
    os.chdir(dataset_dir)
    fu = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]

    folders =os.listdir(dataset_dir)

    files = []
    dataset_csv_file=[]

    for folder in folders:
        for file in os.listdir(folder):
            files.append([dataset_dir+"/"+folder+"/"+file, folder])

        
        df=pd.DataFrame(files, columns=['Image', 'Class']).to_csv('dataset.csv',index=False)
  

    return df
  
if __name__=="__main__":
    csv_file=dataset_preparation()
