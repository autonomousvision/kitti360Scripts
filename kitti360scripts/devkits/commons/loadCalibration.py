# Utils to load transformation to camera pose to system pose
import os
import numpy as np

def checkfile(filename):
    if not os.path.isfile(filename):
        raise RuntimeError('%s does not exist!' % filename)

def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)
    
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None
    
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def loadCalibrationCameraToPose(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename,'r');
     
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
      
    # close file
    fid.close()
    return Tr
    

def loadCalibrationRigid(filename):
    # check file
    checkfile(filename)

    lastrow = np.array([0,0,0,1]).reshape(1,4)
    return np.concatenate((np.loadtxt(filename).reshape(3,4), lastrow))


def loadPerspectiveIntrinsic(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename,'r');

    # read variables
    Tr = {}
    intrinsics = ['P_rect_00', 'R_rect_00', 'P_rect_01', 'R_rect_01']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for intrinsic in intrinsics:
        if intrinsic.startswith('P_rect'):
            Tr[intrinsic] = np.concatenate((readVariable(fid, intrinsic, 3, 4), lastrow))
        else:
            Tr[intrinsic] = readVariable(fid, intrinsic, 3, 3)

    # close file
    fid.close()

    return Tr

if __name__=='__main__':
    
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
                            os.path.realpath(__file__)), '..', '..')

    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    Tr = loadCalibrationCameraToPose(fileCameraToPose)
    print('Loaded %s' % fileCameraToPose)
    print(Tr)

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    Tr = loadCalibrationRigid(fileCameraToVelo)
    print('Loaded %s' % fileCameraToVelo)
    print(Tr)

    fileSickToVelo = os.path.join(kitti360Path, 'calibration', 'calib_sick_to_velo.txt')
    Tr = loadCalibrationRigid(fileSickToVelo)
    print('Loaded %s' % fileSickToVelo)
    print(Tr)

    filePersIntrinsic = os.path.join(kitti360Path, 'calibration', 'perspective.txt')
    Tr = loadPerspectiveIntrinsic(filePersIntrinsic)
    print('Loaded %s' % filePersIntrinsic)
    print(Tr)
