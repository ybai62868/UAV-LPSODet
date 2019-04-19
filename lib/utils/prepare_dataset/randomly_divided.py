from __future__ import print_function
import os
import os.path as osp
import glob
import random



def divide_cls(image_path_lst, train_set_cls, train_Set_lst, valid_set_lst):
    ratio = 0.8 # 8:2
    image_path_size = len(image_path_lst)
    train_set = image_path_lst[:int(ratio * image_path_size)]
    valid_set = image_path_lst[int(ratio * image_path_size):]

    with open(train_set_cls, 'w') as f:
        f.write('\n'.join(train_set))
    
    for valid_path in valid_set:
        assert valid_path not in train_set


def load_data(data_dir, cls):
    return glob.glob(osp.join(data_dir, cls, '*.jpg'))


def shuffle_data(data):
    random.shuffle(data)
    return data


def main():
    TRAIN_DATA_ROOT = '../../../'
    train_set_cls_meta = osp.join(TRAIN_DATA_ROOT, 'dataset', 'train_dataset_cls.txt')
    train_set_meta = osp.join(TRAIN_DATA_ROOT, 'dataset', 'train_dataset.txt')
    valid_set_meta = osp.join(TRAIN_DATA_ROOT, 'dataset', 'valid_dataset.txt')
    valid_set_label_meta = osp.join(TRAIN_DATA_ROOT, 'dataset', 'valid_dataset_label.txt')
    
    dataset_root = osp.join(TRAIN_DATA_ROOT, 'dataset')
    if not osp.exists(dataset_root):
        os.makedirs(dataset_root)
    
    data_dir = osp.join(dataset_root, 'train_dataset')
    cls_lst = sorted(os.listdir(data_dir))
    train_set_cls_lst, train_set_lst, valid_set_lst = [], [], []
    for cls in cls_lst:
        print ('Processing class %s ... ' % (cls))
        train_set_cls = osp.join(data_dir, cls, 'train_%s.txt' % cls)
        train_set_cls_lst.append(train_set_cls)
        image_path_lst = shuffle_data(load_data(data_dir, cls))
        divide_cls(image_path_lst, train_set_cls, train_set_cls, valid_set_lst)
    
    with open(train_set_cls_meta, 'w') as f:
        f.write('\n'.join(train_set_cls_lst))
    
    train_set_lst = shuffle_data(train_set_lst)
    with open(train_set_meta, 'w') as f:
        f.write('\n'.join(train_set_lst))
    with open(valid_set_meta, 'w') as f:
        f.write('\n'.join(valid_set_lst))
    
    label_dict = {}
    for cls in cls_lst:
        label_file = osp.join(dataset_root, 'train_dataset', cls, 'label_all.txt')
        with open(label_file, 'r') as f:
            for line in f:
                lst = line.strip().split()
                frame, box = lst[0], ' '.join(lst[1:])
                label_dict[frame] = box
    label_lst = []
    for frame_path in valid_set_lst:
        frame = frame_path.split(osp.sep)[-1].split('.')[0]
        label_lst.append(label_dict[frame])
    with open(valid_set_label_meta, 'w') as f:
        f.write('\n'.join(label_lst))



if __name__ == '__main__':
    main()