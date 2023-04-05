'''
    Sample Run:
    python prepare_train_val_windows.py

    Converts KITTI-360 labels to windows in numpy format.
    Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, numpy as np, multiprocessing as mp, torch, json, argparse
from kitti360scripts.helpers.labels import id2label
from kitti360scripts.helpers.ply import read_ply
from kitti360scripts.helpers.annotation import Annotation3D
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))

if 'KITTI360_DATASET' in os.environ:
    kitti360Path = os.environ['KITTI360_DATASET']
else:
    kitti360Path = os.path.join(os.path.dirname(
                        os.path.realpath(__file__)), '..', '..')

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split      = opt.data_split
outputPath = os.path.join('.', split)
os.makedirs(outputPath, exist_ok=True)
split_path = 'test1' if split=='test' else split
files      = sorted(glob.glob(os.path.join(kitti360Path, 'data_3d_semantics', split_path, '*', 'static', '*.ply')))

print('================================================================================')
print('data split:    {}'.format(split))
print('output_folder: {}'.format(outputPath))
print('input_folder : {}'.format(os.path.join(kitti360Path, 'data_3d_semantics', split_path)))
print('Found %d ply files' % len(files))
print('================================================================================')

annotation3D_all = {}

def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-15] + '_inst_nostuff.pth')
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')


def f(fn):
    #print(fn)

    data = read_ply(fn)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    points_center = points[:, :3].mean(0)
    coords = np.ascontiguousarray(points[:, :3] - points_center)
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    colors = np.ascontiguousarray(colors) / 127.5 - 1
    colors = colors.astype(np.float32)

    '''semantic'''
    ignore_label = -100
    sem_labels_raw = data['semantic']
    sem_labels = np.ones_like(sem_labels_raw) * ignore_label
    for i in np.unique(sem_labels_raw):
        sem_labels[sem_labels_raw==i] = id2label[i].trainId
    sem_labels[sem_labels==255] = ignore_label
        
    '''instance'''
    instance_labels_raw = data['instance']
    instance_labels = np.ones_like(instance_labels_raw) * ignore_label
    # unique instance id (regardless of semantic label)
    ins_cnt = 0
    ins_map = {}
    for i, ins_id in enumerate(np.unique(instance_labels_raw)):
        if ins_id%1000==0:
            instance_labels[instance_labels_raw==ins_id] = ignore_label
        else:
            instance_labels[instance_labels_raw==ins_id] = ins_cnt
            ins_map[ins_id] = ins_cnt
            ins_cnt+=1
    instance_labels[sem_labels==ignore_label] = ignore_label
    
    '''bounding box'''
    seq = [s for s in fn.split('/') if s.startswith('2013_05_28_drive_')]
    assert(len(seq)==1)
    sequence = seq[0]
    if not sequence  in annotation3D_all.keys():
        label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
        annotation3D = Annotation3D(label3DBboxPath, sequence)
        annotation3D_all[sequence] = annotation3D
    else:
        annotation3D = annotation3D_all[sequence]
    bboxes = []
    for i, ins_id in enumerate(np.unique(instance_labels_raw)):
        if ins_id%1000==0:
            continue
        if id2label[ins_id//1000].trainId==255: # ignored class
            continue
        try:
            obj3D = annotation3D(ins_id//1000, ins_id%1000)
            scale = np.linalg.norm(obj3D.R, axis=0)
            center = obj3D.T# - points_center

            # The object frame of reference is X right, Z up and Y inside
            # The heading angle (\theta) is positive ANTI-clockwise from positive X-axis about the Z-axis.
            # The rotation matrix rot_z is from
            # https://github.com/autonomousvision/kitti360Scripts/blob/e0e3442991d3cf4c69debb84b48fcab3aabf2516/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L186-L192
            # If \theta = 90, rotation matrix rot_z transforms the point (l/2,0) to (0,l/2).
            #                    +Y
            #                    |
            #                    |
            #                    |
            #                    |
            #                    |        (l/2, 0)
            #                    |**********--------------+ X
            #
            #                    + Y
            #                    |
            #                    * (0,l/2)
            #                    *
            #                    *
            #                    *
            #                    *------------------------+ X
            # This example confirms that heading angle is positive anti-clockwise from positive X-axis.

            # Vertices are in the KITTI-360 annotation format:
            # x_corners = [l/2, l/2, l/2, l/2,-l/2,-l/2,-l/2,-l/2]
            # y_corners = [w/2, w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2]
            # z_corners = [h/2,-h/2, h/2,-h/2,-h/2, h/2,-h/2, h/2]
            #
            #                     +Y
            #                     |
            #                     |
            # 5 (-l/2,  w/2)      |             0 (l/2, w/2)
            #           ==========|===========
            #          |          |           |
            # ---------|----------|-----------|-----------+ X
            #          |          |           |
            #           ==========|===========
            # 7 (-l/2, -w/2)      |            2 (l/2, -w/2)
            #                     |
            #                     |
            #
            # NOTE: The KITTI-360 annotation format is different from the
            # the KITTI-360 evaluation format listed here
            # https://github.com/autonomousvision/kitti360Scripts/blob/e0e3442991d3cf4c69debb84b48fcab3aabf2516/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L549-L551
            #
            # To get the heading angle, we first take the midpoint of vertices 0 and 2 with box
            # centered at origin and take the arctan2 of the y-coordinate and x-coordinate
            midpoint            = np.array([ 0.5*(obj3D.vertices[2,0] + obj3D.vertices[0,0]) - center[0], 0.5*(obj3D.vertices[2,1] + obj3D.vertices[0,1]) - center[1] ])
            heading_angle       = np.arctan2(midpoint[1], midpoint[0])
            bboxes.append(([*center, *scale, heading_angle, id2label[ins_id//1000].id, ins_map[ins_id]]))
        except:
            print('Warning: error loading %d!' % ins_id)
            continue

    bboxes = np.array(bboxes)

    # save point center to recover the global coordinate
    seq_id = seq[0].split('_')[-2]
    output_fn = os.path.join(outputPath, '%s_'%seq_id + os.path.basename(fn).replace('.ply', '')+'_center.pth')
    # print('Saving to ' + output_fn)
    # torch.save(points_center, output_fn)

    # save bbox
    output_fn = os.path.join(outputPath, '%s_'%seq_id + os.path.basename(fn).replace('.ply', '')+'.pth')
    #print('Saving to ' + output_fn)
    #torch.save(bboxes, output_fn)
    output_fn = output_fn.replace(".pth", ".npy")
    print('Saving to ' + output_fn)
    np.save(output_fn, bboxes)

    # save coords and labels
    output_fn = os.path.join(outputPath, '%s_'%seq_id + os.path.basename(fn).replace('.ply', '')+'_inst_nostuff.pth')
    # print('Saving to ' + output_fn)
    # torch.save((coords, colors, sem_labels, instance_labels), output_fn)


if __name__=="__main__":
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']

    for fn in files:
        f(fn)
