import os
import cv2
import xml.etree.ElementTree as etxml
import numpy as np
import random
import pickle


class Dataset(object):
    def __init__(self, batch_size=16, img_size=300, saved_pickle=r'dataset',
                 data_dir=(r'D:\datasets_noprevlige\voc2012',),
                 format=('voc',), nclass=21,
                 train_val_test_split=(0.8, 0.1, 0.1), img_preprocess_fn=None) -> None:
        super().__init__()
        self.img_size = img_size
        self.img_preprocess_fn = img_preprocess_fn
        self.train_idx, self.val_idx, self.test_idx = 0, 0, 0
        self.labels = ['background']
        if not os.path.exists(os.path.join(saved_pickle, 'saved_paths.pickle')):
            self.file_lists = []
            self.anno_lists = []
            assert len(data_dir) == len(format)
            for idx in range(len(data_dir)):
                parent = data_dir[idx]
                if format[idx] is 'voc':
                    self.labels += ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                    'cow',
                                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                                    'sofa',
                                    'train', 'tvmonitor']
                    listdir = os.listdir(os.path.join(parent, 'JPEGImages'))
                    self.file_lists += [os.path.join(os.path.join(parent, 'JPEGImages'), _) for _ in listdir]
                    self.anno_lists += [os.path.join(os.path.join(parent, 'Annotations', _.replace('.jpg', '.xml'))) for
                                        _ in listdir]
            assert len(self.file_lists) == len(self.anno_lists)
            temp = list(zip(self.file_lists, self.anno_lists))
            random.shuffle(temp)
            self.file_lists, self.anno_lists = list(zip(*temp))
            self.file_lists = list(self.file_lists)
            self.anno_lists = list(self.anno_lists)
            with open(os.path.join(saved_pickle, 'saved_paths.pickle'), 'wb') as f:
                pickle.dump([self.file_lists, self.anno_lists, self.labels], f)
        else:
            with open(os.path.join(saved_pickle, 'saved_paths.pickle'), 'rb') as f:
                self.file_lists, self.anno_lists, self.labels = pickle.load(f)

        x_train, y_train, x_val, y_val, x_test, y_test = self.split_data(self.file_lists, self.anno_lists,
                                                                         train_val_test_split)

        self.x_train = self.batched(x_train, batch_size)
        self.y_train = self.batched(y_train, batch_size)
        self.x_val = self.batched(x_val, batch_size)
        self.y_val = self.batched(y_val, batch_size)
        self.x_test = self.batched(x_test, batch_size)
        self.y_test = self.batched(y_test, batch_size)

    def split_data(self, x, y, train_val_test_split=(0.8, 0.1, 0.1)):
        assert len(x) == len(y)
        length = len(x)
        train_start, train_end = 0, int(length * train_val_test_split[0])
        val_start, val_end = train_end, int(length * (train_val_test_split[0] + train_val_test_split[1]))
        test_start, test_end = val_end, length
        x_train = x[train_start: train_end]
        y_train = y[train_start: train_end]

        x_val = x[val_start: val_end]
        y_val = y[val_start: val_end]

        x_test = x[test_start: test_end]
        y_test = y[test_start: test_end]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def batched(self, x, batch_size, extend=True):
        length = len(x)
        if not extend or length % batch_size == 0:
            train_length = batch_size * (length // batch_size)
            return np.reshape(x[:train_length], (-1, batch_size))

        target_len = batch_size * (1 + length // batch_size)
        diff = target_len - length
        makeup_list = x[:diff]
        x = x + makeup_list
        return np.reshape(x, newshape=(-1, batch_size))

    def get_actual_data_from_xml(self, xml_path):
        actual_item = []
        try:
            annotation_node = etxml.parse(xml_path).getroot()
            img_width = float(annotation_node.find('size').find('width').text.strip())
            img_height = float(annotation_node.find('size').find('height').text.strip())
            object_node_list = annotation_node.findall('object')
            for obj_node in object_node_list:
                lable = self.labels.index(obj_node.find('name').text.strip())
                bndbox = obj_node.find('bndbox')
                x_min = float(bndbox.find('xmin').text.strip())
                y_min = float(bndbox.find('ymin').text.strip())
                x_max = float(bndbox.find('xmax').text.strip())
                y_max = float(bndbox.find('ymax').text.strip())
                # 位置数据用比例来表示，格式[center_x,center_y,width,height,lable]
                actual_item.append([((x_min + x_max) / 2 / img_width), ((y_min + y_max) / 2 / img_height),
                                    ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
            return actual_item
        except Exception as e:
            print(e)
            return None

    def get_real_data_from_txt(self, txt_path):
        ret = []
        try:
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        if len(l)==0:
                            continue
                        label, x, y, w, h = l.replace('\n', '').split()
                        label = int(label)
                        x = float(x)
                        y = float(y)
                        w = float(w)
                        h = float(h)
                        ret.append([x, y, w, h, label])
                return ret
        except:
            return None

    def get_current_batch(self, x, y, idx):
        return x[idx], y[idx]

    def next_train(self):
        while True:
            try:
                x, y = self.get_current_batch(self.x_train, self.y_train, self.train_idx)
                self.train_idx += 1
                yield self.preprocess(x, y)
            except IndexError:
                self.train_idx = 0
                continue

    def next_val(self):
        while True:
            try:
                x, y = self.get_current_batch(self.x_val, self.y_val, self.val_idx)
                self.val_idx += 1
                yield self.preprocess(x, y)
            except IndexError:
                self.val_idx = 0
                continue

    def next_test(self):
        while True:
            try:
                x, y = self.get_current_batch(self.x_test, self.y_test, self.test_idx)
                self.test_idx += 1
                yield self.preprocess(x, y)
            except IndexError:
                self.test_idx = 0
                continue

    def preprocess(self, x, y):
        assert len(x) == len(y)
        img_buffer = np.zeros(shape=(len(x), self.img_size, self.img_size, 3), dtype=np.float32)
        anno_buffer = []
        for i in range(len(x)):
            img_path = x[i]
            anno_path = y[i]
            img = self.imread(img_path)
            if self.img_preprocess_fn is not None:
                img = self.img_preprocess_fn(img)
            img_buffer[i] = img
            if anno_path.endswith('.xml'):
                anno_item = self.get_actual_data_from_xml(anno_path)
            elif anno_path.endswith('.txt'):
                anno_item = self.get_real_data_from_txt(anno_path)
            anno_buffer.append(anno_item)
        return img_buffer, anno_buffer, x

    def imread(self, path, resize=True):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, (self.img_size, self.img_size))
        return img


if __name__ == '__main__':
    data = Dataset()
    x, y, file_lists = next(data.next_train())
    pass
