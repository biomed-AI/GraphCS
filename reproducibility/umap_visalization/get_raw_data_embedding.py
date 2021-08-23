import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA


def get_raw_embedding(datastr):
    base = "../../data/"
    batch1_label = base + datastr + "/Label1.csv"
    batch2_label = base + datastr + "/Label2.csv"
    label1 = pd.read_csv(batch1_label, header=0).values.flatten()
    label2 = pd.read_csv(batch2_label, header=0).values.flatten()
    labels_x = np.concatenate((label1, label2), axis=0)
    ori_features = np.load(base + datastr + "_feat.npy")
    pca = PCA(n_components=20)
    features_tmp = pca.fit_transform(ori_features)

    save_base = "./Raw_data/"
    save_path = save_base + datastr
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    pd.DataFrame(features_tmp).to_csv(save_path + "/embedding_data.csv")
    pd.DataFrame(labels_x).to_csv(save_path + "/Label.csv", index=False)



names=[

'Baron_mouse_combination',
'Baron_mouse_segerstolpe',
'Baron_mouse_Baron_human'

]


for name in names:
    get_raw_embedding(name)
