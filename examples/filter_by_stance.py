import numpy as np
import pandas as pd
import csv

def write_csv(res, template_path, file):
    csv_file = pd.read_csv(template_path)

    for i in range(20):
        for j in range(300):
            csv_file.iloc[[i], [j+1]] = "news_%06d"%(res[i][j]+1)

    csv_file.to_csv(file, index=False)

template_path = "template.csv"
file = "preds_stance.csv"

stance_path = "preds_stance.npy"
lm_idx_path = "lm_preds_idx.npy"

top_nums = 3000
stance = np.load(stance_path)
lm_idx = np.load(lm_idx_path)

res = []
for i in range(20):
    res_row = []
    for j in range(top_nums):
        idx = i* top_nums + j
        if stance[idx][1] > stance[idx][0]:
            res_row.append(lm_idx[i][j])

    if len(res_row) < 300:
        print("line %d has only %d positive"%(i+1, len(res_row)))
        # res_row = lm_idx[i][:300]
        for j in range(top_nums):
            idx = i* top_nums + j
            if stance[idx][1] < stance[idx][0]:
                res_row.append(lm_idx[i][j])

    res.append(res_row)
    print(len(res_row))

write_csv(res, template_path, file)