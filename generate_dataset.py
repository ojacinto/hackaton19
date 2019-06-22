import numpy as np
import pickle
import os
import re
from xml_parser import open_xml, parser_xml
from image_process import open_image, array_image, crop_image, resize_image, process_image_file

output_filename = "fruit_dataset_test"


def load_classifiers(pickle_name):
    pickle_in = open(pickle_name, "rb")
    loaded = pickle.load(pickle_in)
    pickle_in.close()
    return loaded


def save_classifier(model, file_name='best_classifier'):
    pickle_out = open(file_name, "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


def iterate_files(rootDir="./test"):
    dataset = {
        "data": [],
        'target': []
    }
    for dirName, subdirList, fileList in os.walk(rootDir):
        image, xml = "", ""
        for fname in fileList:
            # Processing name's file
            is_jpg = re.search(".jpg$", fname)
            if is_jpg:
                image = fname
                aux = fname.split('.')
                xml = aux[0] + ".xml"
                is_jpg = False
                root = open_xml(os.path.join(rootDir, xml))
                xml_data = parser_xml(root)
                full_image = open_image(os.path.join(rootDir, image))
                for obj in xml_data[1]:
                    xmin = obj["xmin"]
                    ymin = obj["ymin"]
                    xmax = obj["xmax"]
                    ymax = obj["ymax"]
                    name = obj["name"]
                    fruit = crop_image(full_image, [xmin, ymin, xmax, ymax])
                    fruit = resize_image(fruit, [416, 416])
                    image_data = array_image(fruit)
                    dataset['data'].append(image_data)
                    dataset['target'].append(name)
    dataset['data'] = dataset['data']
    dataset['target'] = dataset['target']
    return dataset


def increment_dataset(rootDir="./test", label='apple'):
    # apple, banana, orange
    dataset = load_classifiers('fruit_dataset_train')
    for dirName, subdirList, fileList in os.walk(rootDir):
        image, xml = "", ""
        for fname in fileList:
            # Processing name's file
            is_jpg = re.search(".jpg$", fname)
            if is_jpg:
                image = fname
                full_image = os.path.join(rootDir, image)
                image_array = process_image_file(full_image)
                dataset['data'].append(image_array)
                dataset['target'].append(label)

    pickle_out = open('fruit_dataset_train_inc', "wb")
    pickle.dump(dataset, pickle_out)


fullpath = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(fullpath, 'train')
dataset = iterate_files(rootDir)
pickle_out = open('fruit_dataset_train', "wb")
pickle.dump(dataset, pickle_out)
pickle_out.close()


