from __future__ import print_function

import os
import os.path as osp
import glob
import shutil
import xml.dom.minidom


def generate_cls(src_path, des_path, cls, idx):
    frame_list = [
        frame.split(osp.sep)[-1].split('.')[0]
        for frame in glob.glob(src_path + '/*.xml')
    ]

    label_all_file = 'label_all.txt'
    label_root = osp.join(des_path, label_all_file)
    with open(label_root, 'w') as fout:
        pass

    cnt = 0
    for frame in frame_list:
        print ('Processing frame %06d of class "%s" ... ' % (cnt, cls))
        image_src = osp.join(src_path, frame + '.jpg')
        label_src = osp.join(src_path, frame + '.xml')

        tr = xml.dom.minidom.parse(label_src)
        width = float(tr.getElementsByTagName('width')[0].firstChild.data)
        height = float(tr.getElementsByTagName('height')[0].firstChild.data)
        xmin = tr.getElementsByTagName('xmin')
        if xmin:
            xmin = float(xmin[0].firstChild.data)
        else:
            continue
        ymin = tr.getElementsByTagName('ymin')
        if ymin:
            ymin = float(ymin[0].firstChild.data)
        else:
            continue

        xmax = tr.getElementsByTagName('xmax')
        if xmax:
            xmax = float(xmax[0].firstChild.data)
        else:
            continue
        
        ymax = tr.getElementsByTagName('ymax')
        if ymax:
            ymax = float(ymax[0].firstChild.data)
        else:
            continue
        
        assert xmin < xmax
        assert ymin < ymax


        cx = (xmin + xmax) / 2.0 / height
        cy = (ymin + ymax) / 2.0 / width
        w  = (xmax - xmin) / width
        h  = (ymax - ymin) / height
        frame_label_file = '%s_%06d.txt' % (cls, cnt)
        frame_label_root = osp.join(des_path, frame_label_file)
        with open(frame_label_root, 'a') as f:
            f.write('%d %f %f %f %f\n' % (idx, cx, cy, w, h))
        
        frame_name = '%s_%06d' % (cls, cnt)
        label_all_root = osp.join(des_path, label_all_file)
        with open(label_all_root, 'a') as f:
            f.write('%s %d %d %d %d\n' % 
                    (frame_name, int(xmin), int(ymin), int(xmax), int(ymax)))
        
        image_des = osp.join(des_path, '%s_%06d.jpg' % (cls, cnt))
        shutil.copyfile(image_src, image_des)

        cnt += 1


def main():
    TRAIN_DATA_ROOT = '../../../'
    src_dir = osp.join(TRAIN_DATA_ROOT, 'data_training')
    des_dir = osp.join(TRAIN_DATA_ROOT, 'train_dataset')
    
    if not osp.exists(des_dir):
        os.makedirs(des_dir)

    cls_lst = sorted(os.listdir(src_dir))
    assert len(cls_lst) == 95
    print (' The number of the sub-class:', len(cls_lst))

    for idx, item in enumerate(cls_lst):
        src_path = osp.join(src_dir, item)
        des_path = osp.join(des_dir, item)
        if not osp.exists(des_path):
            os.makedirs(des_path)

        generate_cls(src_path, des_path, item, idx)



if __name__ == "__main__":
    main()