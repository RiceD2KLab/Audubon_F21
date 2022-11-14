import os
import numpy as np
import pandas as pd
import re
import json

# Grab files
out_dir = os.getcwd() + '/output/Training_models/06_22_bay_tune_retina_2class_set_I_aug/'
mod_dir = out_dir + 'faster_rcnn_R_50_FPN_1x-20221111-122755'

df_parse = None
with open(mod_dir + '/metrics.json') as f:
    lines = f.read().split('\n')
    for i, l in enumerate(lines):
        if l:
            d = json.loads(l)
            if df_parse is None:
                df_parse = pd.DataFrame(columns=list(d.keys()))
            try:
                df_parse.loc[len(df_parse.index)] = list(d.values())
            except:
                print(f'#{i} final results:')
                print(d)

print(list(df_parse))
df_curves = df_parse[['iteration', 'total_loss', 'loss_cls', 'loss_box_reg', 'validation_loss']]

# Plot loss over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4), dpi=200)
iters = df_curves.iteration.values.astype(int)
plt.plot(iters, df_curves.total_loss.values.astype(float), label = 'Train loss (tot)', lw=2)
plt.plot(iters, df_curves.validation_loss.values.astype(float), label = 'Validate loss (tot)', lw=2)
plt.plot(iters, df_curves.loss_cls.values.astype(float), label = 'Train loss (cls)', ls='--', lw=1)
plt.plot(iters, df_curves.loss_box_reg.values.astype(float), label = 'Train loss (box reg)', ls='--', lw=1)

plt.xlabel('Epochs')
plt.ylabel('Losses')
# plt.ylim(0.1, 1.8)

plt.legend()
plt.show()

# Precision Recalls
from detectron2.engine import DefaultPredictor
from utils.evaluation import plot_precision_recall_2, get_precisions_recalls
from utils.hyperparameter import setup1
from utils.dataloader import register_datasets

# Retrieve training parameters
with open(mod_dir + '/parameters.txt') as f:
    cfg_parms = json.loads(f.read())
cfg = setup1(cfg_parms)
cfg.MODEL.WEIGHTS = mod_dir + "/model_final.pth" #"/model_0000898.pth" #"/model_final.pth" # path to the model we just trained
predictor = DefaultPredictor(cfg)

# Register datasets for detectron2
data_dir = "./data/22F/split/" # NOTE!! CHANGE TO YOUR DATA DIR!!
img_ext = '.JPEG'
dirs_full = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if not d.startswith('.') and not d.startswith('_')]
register_datasets(dirs_full, img_ext, cfg_parms['BIRD_SPECIES'], unknown_bird_category=True,
                  bird_species_colors=[(0,0,0)]*len(cfg_parms['BIRD_SPECIES']))

##
print('validation inference:')
val_precisions, val_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_Validate")
plot_precision_recall_2(val_precisions, val_max_recalls, cfg_parms['BIRD_SPECIES'] + ["Unknown Bird"],
                      [(0,0,0)]*(len(cfg_parms['BIRD_SPECIES'])+1))

# print('test inference:')
# test_precisions, test_max_recalls = get_precisions_recalls(cfg, predictor, "birds_species_Test")
# plot_precision_recall_2(test_precisions, test_max_recalls, cfg_parms['BIRD_SPECIES'] + ["Unknown Bird"],
#                       [(0,0,0)]*(len(cfg_parms['BIRD_SPECIES'])+1))

# Confusion matrix

from detectron2.data import MetadataCatalog, DatasetCatalog
from utils.confusion_matrix_birds import confusion_matrix_report


data = DatasetCatalog.get("birds_species_Validate")

# grab the confusion matrix
pred_total, truth_total = confusion_matrix_report(data, predictor,
      cfg_parms['BIRD_SPECIES']+["Unknown Bird"], img_ext='JPEG')

from sklearn.metrics import confusion_matrix, classification_report
from utils.confusion_matrix_birds import plot_confusion_matrix

cm = confusion_matrix(truth_total, pred_total)

fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi=200)
plot_confusion_matrix(cm, cfg_parms['BIRD_SPECIES']+['Missed'], title='',
                      figure=(fig, ax), normalize=False)


