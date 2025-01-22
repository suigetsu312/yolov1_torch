import math
import random
import numpy
import glob
import xml.etree.cElementTree as ET
import matplotlib.image as mpimg

import argparse as arg

class pascal_object:
    def __init__(self, x,y,w,h,category):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.category = category

class pascal_sample:
    def __init__(self, img_name):
        self.img_name = img_name
        self.objects = []
    
class pascal_dataset:
    def __init__(self, dataset_path, size):

        self.classes_name = classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
        self.classes_to_id = dict(zip(classes_name,range(len(classes_name))))
        self.size = size
        self.dataset_path = dataset_path + '\\VOC2012_train_val\\VOC2012_train_val\\Annotations\\'
        self.image_path = dataset_path + '\\VOC2012_train_val\\VOC2012_train_val\\JPEGImages\\'
        self.txt_sample_path = dataset_path 
        self.splited_sample_path = dataset_path
        #save the path of pascal dataset images
        self.xml_sample_path = glob.glob(self.dataset_path + '*.xml')
        self.samples = []

    def extract_objects(self, xml_path):

        tree = ET.parse(xml_path)
        object_node = tree.findall('object')
        img_name = tree.findall('filename')[0].text.split('.')[0]

        cur_sample = pascal_sample(img_name)
        img=mpimg.imread(self.image_path + img_name+'.jpg')
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
        for xml in self.xml_sample_path:
            #preprocessing xml 
            self.extract_objects(xml)

        print('總共 {} 張樣本'.format(len(self.samples)))

    def __write2txt(self, sample : pascal_sample):
        print(self.txt_sample_path + sample.img_name + '.txt')
        with open(self.txt_sample_path + "/sample/" + sample.img_name + '.txt', 'w') as f:
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

        with open(self.splited_sample_path + '\\' + d_set_name + '.txt', 'w+') as f:
            for sample in d_set:
                f.write(self.image_path + sample.img_name + '.jpg\n')
        
    def split_dataset(self, tr):
        random.shuffle(self.samples)
        train = self.samples[:math.floor(self.get_dataset_len() * tr)]
        test = self.samples[math.floor(self.get_dataset_len() * tr):]
        #train = train[]
        #test = test[]
        self.__save_dataset_txt(train,0)
        self.__save_dataset_txt(test,1)
    
def main():

    parser = arg.ArgumentParser()
    parser.add_argument("--dataset_path", default="D:\\prog\\od\\data", dest="dataset_path", type=str)
    args = parser.parse_args()

    dataset = pascal_dataset(args.dataset_path,[224,224])
    dataset.parser()
    dataset.write2txt()
    dataset.split_dataset(0.7)

if __name__ == "__main__":
    main()