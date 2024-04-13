import glob,os
import shutil
from tqdm import tqdm
from argparse import ArgumentParser


def split_validation_subset(args):
    dst_root = os.path.join(args.src_root, 'image_subset')
    front_types = ['COD_*', 'SOD_*', 'SEG_*']

    for front_type in front_types:
        print("Front Type:", front_type)
        src_images = sorted(glob.glob(os.path.join(args.src_root, 'images', front_type)))
        subset_path = os.path.join(dst_root, front_type[:3])
        if not os.path.exists(subset_path):
            os.makedirs(subset_path)
            print('PROCESS', front_type)
            for img in tqdm(src_images):
                imgname = img.split('/')[-1]
                dst_path = os.path.join(subset_path, imgname)
                shutil.copy(img, dst_path)
        else:
            print(subset_path, 'ALREADY EXISTS ...')

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--src_root', default="", help='Path to the test dataset')
    args = parser.parse_args()
    print('Split validation subset ...')
    
    split_validation_subset(args)