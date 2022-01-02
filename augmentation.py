from os import listdir, getcwd
from os.path import isfile, join
import cv2
import albumentations as alb

#root_dir = f'{getcwd()}/datasets/pylons'
root_dir = f'/home/anon/coding/EPT_trees/yolor/datasets/pylons_test'

img_directory_path = f'{root_dir}/images/train'
lbl_directory_path = f'{root_dir}/labels/train'

img_files = [f'{f}' for f in listdir(img_directory_path) if isfile(join(img_directory_path, f))]
label_files = [f'{f}' for f in listdir(lbl_directory_path) if isfile(join(lbl_directory_path, f))]


def get_bboxes(label_file):
    bbox_file = open(label_file, 'r')
    lines = bbox_file.readlines()
    pylon_bboxes_lst = []
    pylon_class_lst = []

    for line in lines:
        bbox = list(map(float, line.split()))
        class_lbl = int(bbox[0])
        pylon_class_lst.append(class_lbl)
        bbox = bbox[1:]
        bbox.insert(4, class_lbl)
        pylon_bboxes_lst.append(bbox)

    return pylon_bboxes_lst, pylon_class_lst


transform_dict = {
    "affine": alb.Affine(scale=1,
                         translate_percent=None,
                         translate_px=0,
                         rotate=90,
                         shear=None,
                         interpolation=1,
                         mask_interpolation=0,
                         cval=0,
                         cval_mask=0,
                         mode=0,
                         fit_output=False,
                         always_apply=True,
                         p=1.0),

    "rain": alb.RandomRain(slant_lower=-3, slant_upper=3, drop_length=5, drop_width=1, drop_color=(200, 200, 200),
                          blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.85),

    "snow": alb.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.5, brightness_coeff=1.2, always_apply=False, p=1.0),
    "crop": alb.Crop(x_min=0, y_min=0, x_max=10, y_max=10, always_apply=False, p=1.0)

}

transform_type = 'affine'
transform = alb.Compose([
    transform_dict[transform_type],
], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))


for index, file in enumerate(img_files):
    image = cv2.imread(f'{img_directory_path}/{img_files[index]}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = get_bboxes(f'{lbl_directory_path}/{label_files[index]}')

    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    bboxes = transformed["bboxes"]

    bboxes_lst = []
    for box in bboxes:
        box_lst = list(box)
        class_label = box_lst[4]
        box_lst = box_lst[:-1]
        box_lst.insert(0, class_label)
        bboxes_lst.append(box_lst)

    image = cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{img_directory_path}/{img_files[index].replace('.png', '')}_{transform_type}.jpg", image)
    with open(f"{lbl_directory_path}/{label_files[index].replace('.png', '')}_{transform_type}.txt", 'w') as trans_lbl_file:
        for bbox in bboxes_lst:
            trans_lbl_file.write(f'{" ".join(map(str, bbox))}\n')

    #print(f'normal: {bboxes}')
    #print(f'\ntransformed: {transformed["bboxes"]}')
   # if index == 0:
   #    break
