import json
from math import sin, cos
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import cv2
from collections import namedtuple
from glob import glob
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from time import time

### Model type data from APOLLOSCAPE dataset

# a label and all meta information
Label = namedtuple('Label', [

    'name'        , # The name of a car type
    'id'          , # id for specific car type
    'category'    , # The name of the car category, 'SUV', 'Sedan' etc
    'categoryId'  , # The ID of car category. Used to create ground truth images
                    # on category level.
    ])


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

models = [
    #     name          id   is_valid  category  categoryId
    Label(             'baojun-310-2017',          0,       '2x',          0),
    Label(                'biaozhi-3008',          1,       '2x',          0),
    Label(          'biaozhi-liangxiang',          2,       '2x',          0),
    Label(           'bieke-yinglang-XT',          3,       '2x',          0),
    Label(                'biyadi-2x-F0',          4,       '2x',          0),
    Label(               'changanbenben',          5,       '2x',          0),
    Label(                'dongfeng-DS5',          6,       '2x',          0),
    Label(                     'feiyate',          7,       '2x',          0),
    Label(         'fengtian-liangxiang',          8,       '2x',          0),
    Label(                'fengtian-MPV',          9,       '2x',          0),
    Label(           'jilixiongmao-2015',         10,       '2x',          0),
    Label(           'lingmu-aotuo-2009',         11,       '2x',          0),
    Label(                'lingmu-swift',         12,       '2x',          0),
    Label(             'lingmu-SX4-2012',         13,       '2x',          0),
    Label(              'sikeda-jingrui',         14,       '2x',          0),
    Label(        'fengtian-weichi-2006',         15,       '3x',          1),
    Label(                   '037-CAR02',         16,       '3x',          1),
    Label(                     'aodi-a6',         17,       '3x',          1),
    Label(                   'baoma-330',         18,       '3x',          1),
    Label(                   'baoma-530',         19,       '3x',          1),
    Label(            'baoshijie-paoche',         20,       '3x',          1),
    Label(             'bentian-fengfan',         21,       '3x',          1),
    Label(                 'biaozhi-408',         22,       '3x',          1),
    Label(                 'biaozhi-508',         23,       '3x',          1),
    Label(                'bieke-kaiyue',         24,       '3x',          1),
    Label(                        'fute',         25,       '3x',          1),
    Label(                     'haima-3',         26,       '3x',          1),
    Label(               'kaidilake-CTS',         27,       '3x',          1),
    Label(                   'leikesasi',         28,       '3x',          1),
    Label(               'mazida-6-2015',         29,       '3x',          1),
    Label(                  'MG-GT-2015',         30,       '3x',          1),
    Label(                       'oubao',         31,       '3x',          1),
    Label(                        'qiya',         32,       '3x',          1),
    Label(                 'rongwei-750',         33,       '3x',          1),
    Label(                  'supai-2016',         34,       '3x',          1),
    Label(             'xiandai-suonata',         35,       '3x',          1),
    Label(            'yiqi-benteng-b50',         36,       '3x',          1),
    Label(                       'bieke',         37,       '3x',          1),
    Label(                   'biyadi-F3',         38,       '3x',          1),
    Label(                  'biyadi-qin',         39,       '3x',          1),
    Label(                     'dazhong',         40,       '3x',          1),
    Label(              'dazhongmaiteng',         41,       '3x',          1),
    Label(                    'dihao-EV',         42,       '3x',          1),
    Label(      'dongfeng-xuetielong-C6',         43,       '3x',          1),
    Label(     'dongnan-V3-lingyue-2011',         44,       '3x',          1),
    Label(    'dongfeng-yulong-naruijie',         45,      'SUV',          2),
    Label(                     '019-SUV',         46,      'SUV',          2),
    Label(                   '036-CAR01',         47,      'SUV',          2),
    Label(                 'aodi-Q7-SUV',         48,      'SUV',          2),
    Label(                  'baojun-510',         49,      'SUV',          2),
    Label(                    'baoma-X5',         50,      'SUV',          2),
    Label(             'baoshijie-kayan',         51,      'SUV',          2),
    Label(             'beiqi-huansu-H3',         52,      'SUV',          2),
    Label(              'benchi-GLK-300',         53,      'SUV',          2),
    Label(                'benchi-ML500',         54,      'SUV',          2),
    Label(         'fengtian-puladuo-06',         55,      'SUV',          2),
    Label(            'fengtian-SUV-gai',         56,      'SUV',          2),
    Label(    'guangqi-chuanqi-GS4-2015',         57,      'SUV',          2),
    Label(        'jianghuai-ruifeng-S3',         58,      'SUV',          2),
    Label(                  'jili-boyue',         59,      'SUV',          2),
    Label(                      'jipu-3',         60,      'SUV',          2),
    Label(                  'linken-SUV',         61,      'SUV',          2),
    Label(                   'lufeng-X8',         62,      'SUV',          2),
    Label(                 'qirui-ruihu',         63,      'SUV',          2),
    Label(                 'rongwei-RX5',         64,      'SUV',          2),
    Label(             'sanling-oulande',         65,      'SUV',          2),
    Label(                  'sikeda-SUV',         66,      'SUV',          2),
    Label(            'Skoda_Fabia-2011',         67,      'SUV',          2),
    Label(            'xiandai-i25-2016',         68,      'SUV',          2),
    Label(            'yingfeinidi-qx80',         69,      'SUV',          2),
    Label(             'yingfeinidi-SUV',         70,      'SUV',          2),
    Label(                  'benchi-SUR',         71,      'SUV',          2),
    Label(                 'biyadi-tang',         72,      'SUV',          2),
    Label(           'changan-CS35-2012',         73,      'SUV',          2),
    Label(                 'changan-cs5',         74,      'SUV',          2),
    Label(          'changcheng-H6-2016',         75,      'SUV',          2),
    Label(                 'dazhong-SUV',         76,      'SUV',          2),
    Label(     'dongfeng-fengguang-S560',         77,      'SUV',          2),
    Label(       'dongfeng-fengxing-SX6',         78,      'SUV',          2)

]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
car_name2id = {label.name: label for label in models}
car_id2name = {label.id: label for label in models}

# Load a 3D model of a car
def model_type_to_vertices_and_triangles(model_type, kaggle_dataset_dir):
    model_name = car_id2name[model_type].name
    with open(f'{kaggle_dataset_dir}/car_models_json/{model_name}.json') as json_file:
        data = json.load(json_file)
    vertices = np.array(data['vertices'])
    vertices[:, 1] = -vertices[:, 1]
    triangles = np.array(data['faces']) - 1
    return vertices, triangles


# k is camera instrinsic matrix
k = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def draw_obj(image, vertices, triangles, color=(0, 0, 255), colors=None):
    if colors is not None:
        for n, t in enumerate(triangles):
            coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
            colorr = colors[n]
            #print(colorr, type(colorr), type(colorr[0]))
            cv2.fillConvexPoly(image, coord, colorr)
            #cv2.polylines(image, np.int32([coord]), 1, (0,0,255))
    else:
        for t in triangles:
            coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
            cv2.fillConvexPoly(image, coord, color)
            #cv2.polylines(image, np.int32([coord]), 1, (0,0,255))


def normalize_to_256(x):
    return (256*(x-x.min())/(x.max()-x.min()) % 256).astype(int)


def process_image(image_id, kaggle_dataset_dir):
    img_name = image_id
    img = cv2.imread(f'{kaggle_dataset_dir}/train_images/{img_name}.jpg',cv2.COLOR_BGR2RGB)[:,:,::-1]
        
    pred_string = train_csv[train_csv.ImageId == img_name].PredictionString.iloc[0]
    items = pred_string.split(' ')
    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
    
    model_type_mask = np.zeros(img.shape[:-1]) - 1  # CLASS MASK
    overlay = np.zeros_like(img)                    # CRUDE COLORING MASK

    for model_type, yaw, pitch, roll, x, y, z in sorted(zip(model_types, yaws, pitches, rolls, xs, ys, zs), key=lambda foo: foo[-1], reverse=True):
        yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
        model_type = int(model_type)
        # I think the pitch and yaw should be exchanged
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        vertices, triangles = model_type_to_vertices_and_triangles(model_type, kaggle_dataset_dir)
        P = np.ones((vertices.shape[0],vertices.shape[1]+1))
        P[:, :-1] = vertices
        P = P.T
        
        img_cor_points = np.dot(Rt, P)
        points3d = img_cor_points.T
        img_cor_points = np.dot(k, img_cor_points)
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        
        draw_obj(model_type_mask, img_cor_points, triangles, color=model_type)
    
        faces_mid_points = np.array([(vertices[t1]+vertices[t2]+vertices[t3])/3 for t1, t2, t3 in triangles])
        faces_mid_points_new = np.array([(points3d[t1]+points3d[t2]+points3d[t3])/3 for t1, t2, t3 in triangles])
        
        #face_ordering = np.argsort(-faces_mid_points_new[:, 1]) # draw faces on each model from bottom
        face_ordering = np.argsort(-faces_mid_points_new[:, 2])  # draw faces on each model from front
        
        # these are face colours
        h_colors = faces_mid_points[:, 1]
        r_colors = np.arctan2(*faces_mid_points[:, [0,2]].T)
        h_colors = normalize_to_256(h_colors)
        r_colors = normalize_to_256(r_colors)
        colors = np.array([(hc, rc, 0) for hc, rc in zip(h_colors, r_colors)]).astype(int)
        
        triangles = triangles[face_ordering]
        colors = colors[face_ordering].tolist()

        draw_obj(overlay, img_cor_points, triangles, colors=colors)#colors=colors)

    model_type_mask = model_type_mask.astype(int)
        
    overlay = overlay.astype(int)
    height_mask = overlay[:, :, 0]
    angle_mask = overlay[:, :, 1]
    
    output = np.zeros_like(img)
    output[:, :, 0] = model_type_mask
    output[:, :, 1] = height_mask
    output[:, :, 2] = angle_mask
    
    return img, model_type_mask, height_mask, angle_mask, output

def visualize(img, model_type_mask, height_mask, angle_mask, *args):
    no_car_black_mask = np.zeros(model_type_mask.shape+(4,))  # this is RGBA
    no_car_black_mask[:, :, 3] = model_type_mask == -1
    
    fig, axs = plt.subplots(2, 2, figsize = (20, 20))
    axs[0, 0].imshow(img)
    #axs[0, 0].set_title('Axis [0,0]')
    
    axs[0, 1].imshow(model_type_mask)
    axs[0, 1].imshow(no_car_black_mask)    
    #axs[0, 1].set_title('Axis [0,1]')
    
    axs[1, 0].imshow(height_mask)
    axs[1, 0].imshow(no_car_black_mask)    
    #axs[1, 0].set_title('Axis [1,0]')
    
    axs[1, 1].imshow(angle_mask, cmap='hsv')
    axs[1, 1].imshow(no_car_black_mask) 
    #axs[1, 1].set_title('Axis [1,1]')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('kaggle_dataset_path', default='data/kaggle')
    arg_parser.add_argument('mask_folder_path', default='data/kaggle/train_target')
    arg_parser.add_argument('-f', '--force', action='store_true', help='force calculating masks again')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='calculate only 10 masks, for debugging')
    arg_parser.add_argument('-p', '--parallel', action='store_true', help='use all cores')

    args = arg_parser.parse_args()

    N_IMAGES = 10 if args.debug else 1e6

    train_csv = pd.read_csv(f'{args.kaggle_dataset_path}/train.csv')
    os.makedirs(args.mask_folder_path, exist_ok=True)

    def target(train_image_path):
        image_id = os.path.split(train_image_path)[1][:-4]
        if (not args.force) and os.path.exists(f'{args.mask_folder_path}/{image_id}.npy'):
            print(f'skipping {image_id}')
            return
        print(f'processing {image_id}')
        img, model_type_mask, height_mask, angle_mask, output = process_image(image_id, args.kaggle_dataset_path)
        np.save(f'{args.mask_folder_path}/{image_id}', output)

    tic = time()
    images_paths = glob(f'{args.kaggle_dataset_path}/train_images/*.jpg')
    if args.debug:
        images_paths = images_paths[:10]

    if args.parallel:
        print('using all cores')
        with ProcessPoolExecutor() as executor:
            executor.map(target, images_paths)

    else:
        print('using only one core')
        for train_image_path in images_paths:
            target(train_image_path)

    print(f'took: {time()-tic:.2f} seconds')
