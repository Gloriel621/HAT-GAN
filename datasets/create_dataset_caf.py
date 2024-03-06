import os
import shutil
import argparse
import random

clusters = ['0-2','3-6','7-9','10-14','15-19',
            '20-29','30-39','40-49','50-69','70-120']

def get_age_cluster(age):
    for cluster in clusters:
        low, high = map(int, cluster.split('-'))
        if age >= low and age <= high:
            return cluster
    return None

def process_image(src_folder, img_name, age, gender, train_split):
    age_cluster = get_age_cluster(age)
    if age_cluster is not None:
        phase = 'train' if random.random() < train_split else 'test'
        dst_folder = os.path.join(gender, f"{phase}{age_cluster}")
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        dst_path = os.path.join(dst_folder, img_name)
        src_path = os.path.join(src_folder, img_name)
        shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../../Cross-Age-Face-Dataset', help='Location of the Cross Age Face dataset')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of images to allocate for training')
    args = parser.parse_args()

    root_folder = args.folder
    train_split = args.train_split

    gender_folders = ['males', 'females']
    
    for gender in gender_folders:
        gender_path = os.path.join(root_folder, gender)
        if not os.path.exists(gender_path):
            print("no such path")
            continue

        celebrity_folders = [f for f in os.listdir(gender_path) if os.path.isdir(os.path.join(gender_path, f))]
        for celeb_folder in celebrity_folders:
            folder_path = os.path.join(gender_path, celeb_folder)
            img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

            for img_file in img_files:
                age = int(img_file.split('_')[1].split('.')[0])
                print('processing {}'.format(img_file))
                process_image(folder_path, img_file, age, gender, train_split)
