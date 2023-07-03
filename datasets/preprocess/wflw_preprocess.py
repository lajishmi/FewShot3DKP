import os
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
import h5py


def extend_bbox(bbox, dl=0, dt=0, dr=0, db=0):
    '''
    Move bounding box sides by fractions of width/height. Positive values enlarge bbox for all sided.
    e.g. Enlarge height bei 10 percent by moving top:
    extend_bbox(bbox, dt=0.1) -> top_new = top - 0.1 * height
    '''
    l, t, r, b = bbox

    if t > b:
        t, b = b, t
    if l > r:
        l, r = r, l
    h = b - t
    w = r - l
    assert h >= 0
    assert w >= 0

    t_new, b_new = int(t - dt * h), int(b + db * h)
    l_new, r_new = int(l - dl * w), int(r + dr * w)

    return np.array([l_new, t_new, r_new, b_new])


n_parts = 98  # number of keypoints
target_size = 128

edge_idx = np.array([
            *([i, i + 1] for i in range(32)),
            *([i, i + 1] for i in range(33, 41)),
            [33, 41],
            *([i, i + 1] for i in range(42, 50)),
            [42, 50],
            *([i, i + 1] for i in range(51, 54)),
            *([i, i + 1] for i in range(55, 59)),
            *([i, i + 1] for i in range(60, 67)),
            *([i, 96] for i in range(60, 68)),
            [60, 67],
            *([i, i + 1] for i in range(68, 75)),
            *([i, 97] for i in range(68, 76)),
            [68, 75],
            *([i, i + 1] for i in range(76, 87)),
            [76, 87],
            *([i, i + 1] for i in range(88, 95)),
            [88, 95],
        ])
print(edge_idx.shape)

train_img = []
train_landmark = []
with open("../data/wflw_raw/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt") as fi:
    for line in fi:
        line = line.split()
        lm = np.array([float(i) for i in line[:196]]).reshape(n_parts, 2)
        bbox = extend_bbox(extend_bbox([float(i) for i in line[196:200]], db=0.1),
                            dl=0.1, dt=0.1, dr=0.1, db=0.1)
        img = Image.open(os.path.join("../data/wflw_raw/WFLW_images", line[-1])).crop(bbox).convert('RGB')
        img = img.resize((target_size, target_size), resample=Image.BILINEAR)
        img = np.asarray(img).transpose((2, 0, 1))
        lm[:, 0] = (lm[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * 2 - 1
        lm[:, 1] = (lm[:, 1] - bbox[1]) / (bbox[3] - bbox[1]) * 2 - 1
        train_img.append(img)
        train_landmark.append(lm)


test_img = []
test_landmark = []
with open("../data/wflw_raw/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt") as fi:
    for line in fi:
        line = line.split()
        lm = np.array([float(i) for i in line[:196]]).reshape(n_parts, 2)
        bbox = extend_bbox(extend_bbox([float(i) for i in line[196:200]], db=0.1),
                            dl=0.1, dt=0.1, dr=0.1, db=0.1)
        img = Image.open(os.path.join("../data/wflw_raw/WFLW_images", line[-1])).crop(bbox).convert('RGB')
        img = img.resize((target_size, target_size), resample=Image.BILINEAR)
        img = np.asarray(img).transpose((2, 0, 1))
        lm[:, 0] = (lm[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * 2 - 1
        lm[:, 1] = (lm[:, 1] - bbox[1]) / (bbox[3] - bbox[1]) * 2 - 1
        test_img.append(img)
        test_landmark.append(lm)


train_img = np.stack(train_img)
train_landmark = np.stack(train_landmark)
test_img = np.stack(test_img)
test_landmark = np.stack(test_landmark)


file = h5py.File('../data/wflw/wflw.h5', "w")
file.create_dataset("train_img", np.shape(train_img), h5py.h5t.STD_U8BE, data=train_img)
file.create_dataset("train_landmark", np.shape(train_landmark), "float32", data=train_landmark)
file.create_dataset("test_img", np.shape(test_img), h5py.h5t.STD_U8BE, data=test_img)
file.create_dataset("test_landmark", np.shape(test_landmark), "float32", data=test_landmark)
file.create_dataset("edge_idx", np.shape(edge_idx), "int", data=edge_idx)
file.close()