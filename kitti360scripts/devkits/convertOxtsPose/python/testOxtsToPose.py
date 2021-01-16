# Test script for loading Oxts data and convert to Mercator coordinate
import os
from data import loadOxtsData
from utils import postprocessPoses
from convertOxtsToPose import convertOxtsToPose

if __name__=="__main__":

  # root dir of KITTI-360
  if 'KITTI360_DATASET' in os.environ:
      kitti360_dir = os.environ['KITTI360_DATASET']
  else:
      kitti360_dir = os.path.join(os.path.dirname(
                                 os.path.realpath(__file__)), '..', '..')

  # load Oxts data
  seq_id = 0
  oxts_dir = os.path.join(kitti360_dir, 'data_poses', '2013_05_28_drive_%04d_sync'%seq_id, 'oxts')
  if not os.path.isdir(oxts_dir):
    raise ValueError('%s does not exist! \nPlease specify KITTI360_DATASET in your system path.\nPlease check if you have downloaded OXTS poses (data_poses_oxts.zip) and unzipped them under KITTI360_DATASET' % oxts_dir)
  oxts,ts = loadOxtsData(oxts_dir)
  print('Loaded oxts data from %s' % oxts_dir)
  
  # convert to Mercator coordinate
  poses = convertOxtsToPose(oxts)
  
  # convert coordinate system from
  #   x=forward, y=right, z=down 
  # to
  #   x=forward, y=left, z=up
  poses = postprocessPoses(poses)

  # write to file
  output_dir = 'output'
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  output_file = '%s/2013_05_28_drive_%04d_sync_oxts2pose.txt'% (output_dir, seq_id)
  
  with open(output_file, 'w') as f:
      for pose_ in poses:
          pose_ = ' '.join(['%.6f'%x for x in pose_.flatten()])
          f.write('%s\n'%pose_)
  print('Output written to %s' % output_file)
