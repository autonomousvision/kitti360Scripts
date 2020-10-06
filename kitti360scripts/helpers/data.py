import os
import numpy as np

class KITTI360(object):
    def __init__(self, data_dir, seq=0, cam=0):

        if cam!=0:
            raise NotImplementedError('Please generate cam%d_to_world.txt at first!')
    
        # intrinsics
        calib_dir = '%s/calibration' % (data_dir)
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')

        # camera poses 
        sequence_dir = '%s/2013_05_28_drive_%04d_sync/' % (data_dir, seq)
        self.pose_file = os.path.join(sequence_dir, 'cam%d_to_world.txt' % cam)
        self.image_dir = '%s/image_%02d/data_rect/' % (sequence_dir, cam)

        assert os.path.isfile(self.pose_file), '%s does not exist!' % self.pose_file
        assert os.path.isfile(self.intrinsic_file), '%s does not exist!' % self.intrinsic_file
        
        print('-----------------------------------------------')
        print('Loading KITTI-360, sequence %04d, camera %d' % (seq, cam))
        print('-----------------------------------------------')
        self.load_intrinsics()
        print('-----------------------------------------------')
        self.load_poses()
        print('-----------------------------------------------')
        
    def load_intrinsics(self):
        # load intrinsics
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(self.intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            if line[0] == "S_rect_00:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)

        self.K = K
        self.width = width
        self.height = height
        print ('Image size %dx%d ' % (self.width, self.height))
        print ('Intrinsics \n', self.K)

    def load_poses(self):
        # load poses of the current camera
        poses = np.loadtxt(self.pose_file)
        self.frames = poses[:,0].astype(np.int)
        self.poses = np.reshape(poses[:,1:], (-1, 4, 4))
        print('Number of posed frames %d' % len(self.frames))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        pose = self.poses[idx]
        basename = '%010d.png' % frame
        image_file = os.path.join(self.image_dir, basename)
        assert os.path.isfile(image_file), '%s does not exist!' % image_file
        print(pose)
        print(image_file)
        return 



if __name__=='__main__':
    dset = KITTI360('/is/rg/avg/datasets/KITTI360/2013_05_28/')
    for i in range(10):
        dset[i]

