import os
import math
import random
import numpy
import glob
import xml.etree.cElementTree as ET
import matplotlib.image as mpimg
from tools import PathBind
import argparse as arg

class pascal_object:
    def __init__(self, x, y, w, h, category):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.category = category

class pascal_sample:
    def __init__(self, img_name, img_path):
        self.img_name = img_name
        self.img_path = img_path
        self.objects = []

class pascal_dataset:
    def __init__(self, dataset_paths, out_root, size):

        self.classes_name = classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.classes_to_id = dict(zip(classes_name, range(len(classes_name))))
        self.size = size
        # 支援多個 VOC 根目錄（例如 2007/2012），輸出統一到 out_root
        self.sources = []
        for p in dataset_paths:
            root = PathBind(p)
            self.sources.append({
                "ann_root": os.path.join(root, "Annotations"),
                "img_root": os.path.join(root, "JPEGImages"),
            })
        self.txt_sample_path = PathBind(out_root)
        self.splited_sample_path = self.txt_sample_path
        self.samples = []

    def extract_objects(self, xml_path, img_root):

        tree = ET.parse(xml_path)
        object_node = tree.findall('object')
        img_name = tree.findall('filename')[0].text.split('.')[0]

        img_path = PathBind(os.path.join(img_root, img_name + ".jpg"))
        cur_sample = pascal_sample(img_name, img_path)
        
        img=mpimg.imread(img_path)
        y,x,_ = img.shape
        for ob in object_node:
            #get object class
            category = ob.find('name').text
            #get bbox
            bbox = ob.find('bndbox')
            #標準化
            xmin = round(float(bbox.find('xmin').text)/x,6) 
            xmax = round(float(bbox.find('xmax').text)/x,6)
            ymin = round(float(bbox.find('ymin').text)/y,6)
            ymax = round(float(bbox.find('ymax').text)/y,6)
            
            w = (xmax - xmin)
            h = (ymax - ymin)
            x_c = round(float(xmin+w/2),6)
            y_c = round(float(ymin+h/2),6)
            cur_sample.objects.append(pascal_object(x_c,y_c,w,h,category))
        self.samples.append(cur_sample)
        
    def parser(self):
        for src in self.sources:
            xml_list = glob.glob(os.path.join(src["ann_root"], '*.xml'))
            print('開始解析 {} 下的 {} 個 xml 檔'.format(src["ann_root"], len(xml_list)))
            for xml in xml_list:
                # preprocessing xml 
                self.extract_objects(xml, src["img_root"])

        print('總共 {} 張樣本'.format(len(self.samples)))

    def __write2txt(self, sample : pascal_sample):
        cur_path = PathBind(self.txt_sample_path + "/sample/" + sample.img_name + '.txt')
        os.makedirs(os.path.dirname(cur_path), exist_ok=True)
        with open(cur_path, 'w') as f:
            for o in sample.objects:
                f.write('{} {} {} {} {}\n'.format(self.classes_to_id[o.category], o.x, o.y, o.w, o.h))

    def write2txt(self):
        if len(self.samples) == 0:
            return
        print('開始將樣本寫入txt檔')
        for s in self.samples:
            self.__write2txt(s)
        print('結束')

    def get_dataset_len(self):
        return len(self.samples)

    def __save_dataset_txt(self, d_set, mode):
        '''
            mode =0 : train
            mode =1 : test
        '''
        if mode == 0:
            d_set_name = 'train'
        elif mode == 1:
            d_set_name = 'test'
        else:
            raise ValueError(f'{mode} is invaild value, mode must be 0 or 1.')
        cur_path = PathBind(self.splited_sample_path + '/' + d_set_name + '.txt')
        with open(cur_path, 'w+') as f:
            for sample in d_set:
                f.write(PathBind(sample.img_path) + '\n')
        
    def split_dataset(self, tr):
        random.shuffle(self.samples)
        train = self.samples[:math.floor(self.get_dataset_len() * tr)]
        test = self.samples[math.floor(self.get_dataset_len() * tr):]
        self.__save_dataset_txt(train,0)
        self.__save_dataset_txt(test,1)
    
def main():

    parser = arg.ArgumentParser()
    parser.add_argument("--voc2007", default="~/dataset/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007",
                        help="VOC2007 根目錄 (含 Annotations/JPEGImages)")
    parser.add_argument("--voc2012", default= "/home/natsu/dataset/VOC2012/VOC2012_train_val/VOC2012_train_val/",
                        help="VOC2012 根目錄 (含 Annotations/JPEGImages)")
    parser.add_argument("--out-root", default="~/dataset/VOC0712_merged", help="輸出 sample/ 與 train/test.txt 的目錄")
    args = parser.parse_args()

    dataset = pascal_dataset([args.voc2007, args.voc2012], args.out_root, [448,448])
    dataset.parser()
    dataset.write2txt()
    dataset.split_dataset(0.7)

if __name__ == "__main__":
    main()
