import json
import numpy as np


def loadlabel(json_file_labels, json_new_labels, json_longtail):

    label_path = json_file_labels
    with open(label_path, 'r') as jsonfile:
        label_and_id = json.load(jsonfile)
    labelsandid = {}
    for k, v in label_and_id.items():
        labelsandid[v] = k  # ground truth label(dict)  id:label
    del label_and_id

    new_label_path = json_new_labels
    with open(new_label_path, 'r') as jsonfile:
        new_label = json.load(jsonfile)  # vplabel(dict) classid:label,verb,verb_id,prep,prep_id

    longtailclass_path = json_longtail
    with open(longtailclass_path, 'r') as json_file:
        longtail = json.load(json_file)  # longtailclasses' label and text (dict)
    longtaillabel = []
    for i in longtail.keys():
        longtaillabel.append(eval(i))

    longtailclass = []
    for key in longtail.keys():
        # V = new_label[key]['verb_id']
        # P = new_label[key]['prep_id']
        longtailclass.append([new_label[key]['verb_id'], new_label[key]['prep_adv_id']])
    longtailclass = np.array(longtailclass)

    return labelsandid, new_label, longtaillabel, longtailclass
