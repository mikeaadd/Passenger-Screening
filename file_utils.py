import os, shutil, random

def create_dirs(dst):

    # make directory for partitioned data
    if not os.path.exists(dst):
        os.mkdir(dst)

    for end in ['train', 'val', 'test']:
        dir = os.path.join(dst, end)
        if not os.path.exists(dir):
                os.mkdir(dir)

    return None

def move_to_folders(src, dst):

    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    random.shuffle(files)
    n_train = round(len(files)*.8*.8)
    train_files = files[:n_train]
    n_val = round(len(files)*.8*.2)
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    for file in train_files:
        srcf = os.path.join(src, file)
        dstf = os.path.join(dst, 'train', file)
        shutil.copyfile(srcf, dstf)
    final_files = [f for f in os.listdir(os.path.join(dst, 'train')) if os.path.isfile(os.path.join(src, f))]
    print('There are {} files in your Train directory'.format(len(final_files)))

    for file in val_files:
        srcf = os.path.join(src, file)
        dstf = os.path.join(dst, 'val', file)
        shutil.copyfile(srcf, dstf)
    final_files = [f for f in os.listdir(os.path.join(dst, 'val')) if os.path.isfile(os.path.join(src, f))]
    print('There are {} files in your Val directory'.format(len(final_files)))


    for file in test_files:
        srcf = os.path.join(src, file)
        dstf = os.path.join(dst, 'test', file)
        shutil.copyfile(srcf, dstf)
    final_files = [f for f in os.listdir(os.path.join(dst, 'test')) if os.path.isfile(os.path.join(src, f))]
    print('There are {} files in your Test directory'.format(len(final_files)))


if __name__ == '__main__':

    current_path = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(current_path, 'data/stage1_aps')
    dst = os.path.join(current_path, 'data')
    create_dirs(dst)
    move_to_folders(src, dst)
