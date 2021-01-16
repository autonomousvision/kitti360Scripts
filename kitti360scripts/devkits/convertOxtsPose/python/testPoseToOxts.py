# Test script for loading Oxts data and convert to Mercator coordinate
import os
from data import loadPoses
from utils import postprocessPoses
from convertPoseToOxts import convertPoseToOxts

if __name__=="__main__":

  # root dir of KITTI-360
  if 'KITTI360_DATASET' in os.environ:
      kitti360_dir = os.environ['KITTI360_DATASET']
  else:
      kitti360_dir = os.path.join(os.path.dirname(
                                 os.path.realpath(__file__)), '..', '..')

  # load poses
  seq_id = 0

  pose_file = os.path.join(kitti360_dir, 'data_poses', '2013_05_28_drive_%04d_sync'%seq_id, 'poses.txt')
  if not os.path.isfile(pose_file):
    raise ValueError('%s does not exist! \nPlease specify KITTI360_DATASET in your system path.\nPlease check if you have downloaded system poses (data_poses.zip) and unzipped them under KITTI360_DATASET' % pose_file)
  [ts, poses] = loadPoses(pose_file)
  print('Loaded pose file %s' % pose_file)
  
  # convert coordinate system from
  #   x=forward, y=left, z=up
  # to
  #   x=forward, y=right, z=down 
  poses = postprocessPoses(poses)
  
  # convert to lat/lon coordinate
  oxts = convertPoseToOxts(poses)
  
  # write to file
  output_dir = 'output'
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  output_file = '%s/2013_05_28_drive_%04d_sync_pose2oxts.txt'% (output_dir, seq_id)
  with open(output_file, 'w') as f:
      for oxts_ in oxts:
          oxts_ = ' '.join(['%.6f'%x for x in oxts_])
          f.write('%s\n'%oxts_)
  print('Output written to %s' % output_file)
