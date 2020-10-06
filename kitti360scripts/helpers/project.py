import os
import numpy as np

class Projection:
    def __init__(self, K):
        self.K = K

    def project(self, points, R, T, inverse=False):
        assert (points.ndim==R.ndim)
        assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
        ndim=R.ndim
        if ndim==2:
            R = np.expand_dims(R, 0) 
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
        else:
            points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

        if ndim==2:
            points = points[0]

        return points

    def perspective(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.K[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth




        # project 3d points from training frames to test frames
        curr_pose = np.reshape(poses, [-1,4,4])
        T = np.reshape(curr_pose[:, :3, 3], [-1,1,3])
        R = curr_pose[:, :3, :3]

        # convert points from world coordinate to local coordinate 
        points_local = projector.project(vertices, R, T, inverse=True)

        # perspective projection
        u,v,depth = projector.perspective(points_local)


class Camera:
    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync'):
        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        pose_file = os.path.join(pose_dir, "cam0_to_world.txt")
        intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        
        # load intrinsics
        self.load_intrinsics(intrinsic_file)

        # load poses
        poses = np.loadtxt(pose_file)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,4,4])
        self.poses = {}
        for frame, pose in zip(frames, poses): 
            self.poses[frame] = pose

        # projector
        self.projector = Projection(self.K)

    def load_intrinsics(self, intrinsic_file):
    
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
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
        self.width, self.height = width, height

    def project_vertices(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.poses[frameId]
        T = curr_pose[:3,  3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate 
        points_local = self.projector.project(vertices, R, T, inverse)

        # perspective projection
        u,v,depth = self.projector.perspective(points_local)

        return (u,v), depth 

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 
        obj3d.generateMeshes()
