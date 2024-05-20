import argparse
import sys
sys.path.append('GERALD/gerald-tools')
import gerald_tools
import shutil
import os

def main(p: str):
    """
    Simply plots first image and annotations
    :param p: Command line argument for GERALD dataset path
    """
    # gerald_train = gerald_tools.GERALDDataset(p,subset="train")
    # gerald_val = gerald_tools.GERALDDataset(p,subset="val")
    # gerald_test = gerald_tools.GERALDDataset(p,subset="test")
    gerald = gerald_tools.GERALDDataset(p)
    gerald_train = gerald_tools.GERALDDataset(p, subset="train")
    gerald_val = gerald_tools.GERALDDataset(p, subset="val")
    gerald_test = gerald_tools.GERALDDataset(p, subset="test")
    destination_path = "/home/paulo/GERALD_YOLO/"
    # Define the source and destination paths
    for annotation in gerald_test.subset_annotations:
        source_file = os.path.join(p + "/JPEGImages", annotation.src_name)
        destination_file = os.path.join(destination_path + "images/test", annotation.src_name)
        # Use shutil.copy() to copy the file
        shutil.copy(source_file, destination_file)
        src_height = annotation.src_height
        src_width = annotation.src_width
        file_name = annotation.src_name.replace(".jpg",".txt")
        file_path = destination_path + "labels/test/" + file_name
        with open(file_path, 'w') as f:
            for obj in annotation.objects:
                id = obj.label.value
                width = obj.x_max - obj.x_min
                height = obj.y_max - obj.y_min
                center_x = (obj.x_min + width/2)/src_width
                center_y = (obj.y_min + height/2)/src_height
                width = width/src_width
                height = height/src_height
                str_list = map(str, [id, center_x, center_y, width, height])
                # Join the list of strings with a space separator
                row = ' '.join(str_list)
                f.write(row + "\n")
    
    for annotation in gerald_train.subset_annotations:
        source_file = os.path.join(p + "/JPEGImages", annotation.src_name)
        destination_file = os.path.join(destination_path + "images/train", annotation.src_name)
        # Use shutil.copy() to copy the file
        shutil.copy(source_file, destination_file)
        src_height = annotation.src_height
        src_width = annotation.src_width
        file_name = annotation.src_name.replace(".jpg",".txt")
        file_path =  destination_path + "labels/train/" + file_name
        with open(file_path, 'w') as f:
            for obj in annotation.objects:
                id = obj.label.value
                width = obj.x_max - obj.x_min
                height = obj.y_max - obj.y_min
                center_x = (obj.x_min + width/2)/src_width
                center_y = (obj.y_min + height/2)/src_height
                width = width/src_width
                height = height/src_height
                str_list = map(str, [id, center_x, center_y, width, height])
                # Join the list of strings with a space separator
                row = ' '.join(str_list)
                f.write(row + "\n")
    
    for annotation in gerald_val.subset_annotations:
        source_file = os.path.join(p + "/JPEGImages", annotation.src_name)
        destination_file = os.path.join(destination_path + "images/val", annotation.src_name)
        # Use shutil.copy() to copy the file
        shutil.copy(source_file, destination_file)
        src_height = annotation.src_height
        src_width = annotation.src_width
        file_name = annotation.src_name.replace(".jpg",".txt")
        file_path = destination_path + "labels/val/" + file_name
        with open(file_path, 'w') as f:
            for obj in annotation.objects:
                id = obj.label.value
                width = obj.x_max - obj.x_min
                height = obj.y_max - obj.y_min
                center_x = (obj.x_min + width/2)/src_width
                center_y = (obj.y_min + height/2)/src_height
                width = width/src_width
                height = height/src_height
                str_list = map(str, [id, center_x, center_y, width, height])
                # Join the list of strings with a space separator
                row = ' '.join(str_list)
                f.write(row + "\n")

    # im, targets, idx = gerald[0]
    # gerald_tools.plot_targets_over_im(im, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to GERALD dataset")
    args = parser.parse_args()

    sys.exit(main(p="/home/paulo/GERALD/dataset"))
