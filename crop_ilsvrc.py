from scipy.misc import imread, imsave
from glob import glob
import os

root = '/media/david/1600E62300E60997/ILSVRC'
data = '/Data/DET/train/ILSVRC2013_train'
annotations = '/Annotations/DET/train/ILSVRC2013_train'

def extract_tag_value(annotation, tag):
    start_tag = '<'+tag+'>'
    stop_tag = '</'+tag+'>'
    return int(annotation[annotation.index(start_tag)+len(start_tag): annotation.index(stop_tag)])

def parse_annotation(image_path):
    image_category = image_path[image_path.rfind('/')+1: image_path.rfind('_')]
    image_name = image_path[image_path.rfind('_')+1: -5]
    annotation_path = root + annotations + '/' + image_category + '/' + image_category+'_'+image_name+'.xml'
    with open(annotation_path) as annotation_file:
        annotation = annotation_file.read()
    return [extract_tag_value(annotation, tag) for tag in ('xmin', 'xmax', 'ymin', 'ymax')]

data_folders = glob(root+data+'/*')
for i, data_folder in enumerate(data_folders):
    print(i, len(data_folders))
    category = data_folder[data_folder.rfind('/'):]
    subfolder = root+'/cropped'+category
    if not os.path.isdir(subfolder):
        os.mkdir(subfolder)
    annotation_folder = root + annotations + category
    for image_path in glob(root+data+category+'/*'):
        try:
            image_array = imread(image_path)
            xmin, xmax, ymin, ymax = parse_annotation(image_path)
            cropped = image_array[ymin: ymax, xmin: xmax, :]
            image_name = image_path[image_path.rfind('_')+1: -5]
            save_path = root + '/cropped'+ category + category + '_' + image_name + '.jpeg'
            imsave(save_path, cropped)
        except:
            print('error on image_path=', image_path)



