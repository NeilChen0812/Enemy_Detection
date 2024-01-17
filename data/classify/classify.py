import os
import shutil


def copy_and_rename(src_folder, dest_folder1, dest_folder2):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(dest_folder1):
        os.makedirs(dest_folder1)
    if not os.path.exists(dest_folder2):
        os.makedirs(dest_folder2)

    # 获取源文件夹中的所有子文件夹
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]

    for subfolder in subfolders:
        # 从子文件夹路径中获取文件夹名
        folder_name = os.path.basename(subfolder)

        # 构建源文件路径和目标文件路径
        src_file1 = os.path.join(subfolder, 'label.png')
        dest_file1 = os.path.join(dest_folder1, folder_name[:-5] + '.png')

        src_file2 = os.path.join(subfolder, 'img.png')
        dest_file2 = os.path.join(dest_folder2, folder_name[:-5] + '.png')

        # 复制并重命名文件
        shutil.copy2(src_file1, dest_file1)
        shutil.copy2(src_file2, dest_file2)

dir_path = os.path.dirname(os.path.realpath(__file__))
# 设置源文件夹和目标文件夹的路径
source_folder = dir_path + '/labelme_json'
destination_folder1 = dir_path + '/cv2_mask'
destination_folder2 = dir_path + '/pic'

# 执行复制和重命名操作
copy_and_rename(source_folder, destination_folder1, destination_folder2)
