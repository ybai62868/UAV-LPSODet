# two json file.
# 1. for the prediction 
# 2. for the ground truth

import mmcv
import numpy as np
import os
import os.path as osp
import glob

val_root = './data/dac/annotations/val_new_20.json'
pred_root = './results/CornerNet_Squeeze/310000/validation/results.json'


def bbox_iou(box1, box2):
    b1_x1 = box1[0]
    b1_y1 = box1[1]
    b1_x2 = box1[0] + box1[2]
    b1_y2 = box1[1] + box1[3]

    b2_x1 = box2[0]
    b2_y1 = box2[1]
    b2_x2 = box2[0] + box2[2]
    b2_y2 = box2[1] + box2[3]

    inter_rect_x1 = np.max((b1_x1, b2_x1))
    inter_rect_y1 = np.max((b1_y1, b2_y1))
    inter_rect_x2 = np.min((b1_x2, b2_x2))
    inter_rect_y2 = np.min((b1_y2, b2_y2))

    inter_area = 0.0
    if inter_rect_x2 - inter_rect_x1 + 1 > 0:
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

def fun(data_pred):
    res = []
    for i in range(len(data_pred)):
        res.append(data_pred[i]['image_id'])
    res = list(set(res))
    return res

def score_max(data_pred):
    res_rm = []
    #temp = data_pred[0]
    for i in range(len(data_pred) - 1):
        if data_pred[i]['image_id'] == data_pred[i+1]['image_id']:
            if data_pred[i]['score'] < data_pred[i+1]['score']:
                temp = data_pred[i+1]
        else:
            res_rm.append(temp)
        if i + 1 == len(data_pred) - 1:
            res_rm.append(temp)
    return res_rm

def score_max2(data_pred):
    res_rm = []
    for i in range(len(data_pred)-1):
        for j in range(i+1, len(data_pred)):
            if (data_pred[i]['image_id'] == data_pred[j]['image_id']):
                if (data_pred[i]['score'] > data_pred[j]['score']):
                    temp = data_pred[i]
                else:
                    temp = data_pred[j]
            else:
                res_rm.append(temp)


def main():
    data_pred = mmcv.load(pred_root)
    data_gt = mmcv.load(val_root)
    #res_rm = score_max2(data_pred)
    res_rm = score_max(data_pred)
    
    res = np.array([])
    #print (type(data_pred))
    #print (type(data_gt))
    
    res_gt = data_gt['annotations']
    print ('the length of gt', len(res_gt))
    print ('the length of pred', len(res_rm))

    print (len(res_gt), len(res_rm))
    for i in range(len(res_gt)):
        for j in range(len(res_rm)):
            if res_rm[j]['image_id'] == res_gt[i]['image_id']:
                #print ('box1', res_rm[j]['bbox'])
                #print ('box2', res_gt[i]['bbox'])
                iou = bbox_iou(res_rm[j]['bbox'], res_gt[i]['bbox'])
                iou = np.array([iou])
                print ('Current {} iou => {}'.format(res_rm[j]['image_id'], iou))
                res = np.concatenate((res, iou))
                #print ('Current res =>', res)
                break

    print ('Total Average IoU = ', np.mean(res))


if __name__ == '__main__':
    main()
