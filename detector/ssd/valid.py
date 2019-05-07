# two json file.
# 1. for the prediction 
# 2. for the validation

import mmcv
import numpy as np
import os
import os.path as osp
import glob

pred_root = ''
val_root = ''


def bbox_iou(box1, box2):
    b1_x1 = box1[0]
    b1_y1 = box1[1]
    b2_x1 = box1[0] + box1[2]
    b2_y2 = box1[1] + box1[3]

    b2_x1 = box2[0]
    b2_y1 = box2[1]
    b2_x2 = box2[0] + box2[2]
    b2_y2 = box2[1] + box2[3]

    inter_rect_x1 = np.max((b1_x1, b2_x1))
    inter_rect_y1 = np.max((b1_y1, b2_y1))
    inter_rect_x2 = np.min((b1_x2, b2_x2))
    inter_rect_y2 = np.min((b1_y2, b2_y2))

    inter_area = 0.0
    if inter_rect_x1 - inter_rect_x2 + 1 > 0:
        inter_width = inter_rect_x2 - inter_rect_x1 + 1
    else:
        inter_width = 0
    
    if inter_rect_y2 - inter_rect_y1 + 1 > 0:
        inter_height = inter_rect_y2 - inter_rect_y1 + 1
    else:
        inter_height = 0

    inter_area = inter_width * inter_height
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def main():
    data_pred = mmcv.load(pred_root)
    data_gt = mmcv.load(pred_root)
    res = np.array([])
    print (type(data_pred))
    print (type(data_gt))
    
    


    ious = np.concatenate((ious, iou), axis = 0)
    print ('Total Average IoU = ', np.mean(res))



if __name__ == '__main__':
    main()