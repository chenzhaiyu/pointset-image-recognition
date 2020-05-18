"""
Write semantic 3D dataset into S3DIS format
"""

import numpy as np
from tqdm import tqdm
import sys


def partition_train_val_file(file, label_file, out_path, area_id):

    points = np.loadtxt(file)
    labels = np.loadtxt(label_file)

    file_name = file.split('/')[-1]
    room_type = file_name.split('_')[1]

    num_points = points.shape[0]
    room_size = int(np.floor(num_points / 100))
    for room_id in tqdm(range(100)):
        room = points[room_id * room_size: (room_id + 1) * room_size, [0, 1, 2, 4, 5, 6]]
        room_labels = labels[room_id * room_size: (room_id + 1) * room_size]
        room = np.concatenate((room, room_labels.reshape((-1, 1))), axis=1)
        np.save(out_path + '/Area_{}_'.format(area_id) + room_type + '_{}.npy'.format(room_id+1), room)
        pass


if __name__ == '__main__':

    partition_train_val_file(sys.argv[1],
                             sys.argv[2],
                             sys.argv[3], int(sys.argv[4]))   # train file: area_id 1
