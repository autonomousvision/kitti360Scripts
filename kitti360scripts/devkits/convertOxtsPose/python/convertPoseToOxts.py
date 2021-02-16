import numpy as np
from utils import latToScale, latlonToMercator, mercatorToLatlon

def convertPoseToOxts(pose):
  '''converts a list of metric poses into oxts measurements,
  starting at (0,0,0) meters, OXTS coordinates are defined as
  x = forward, y = right, z = down (see OXTS RT3000 user manual)
  afterwards, pose{i} contains the transformation which takes a
  3D point in the i'th frame and projects it into the oxts
  coordinates of the first frame.'''

  single_value = not isinstance(pose, list)
  if single_value:
    pose = [pose]
  
  # origin in OXTS coordinate
  origin_oxts = [48.9843445, 8.4295857] # lake in Karlsruhe
  
  # compute scale from lat value of the origin
  scale = latToScale(origin_oxts[0])
  
  # origin in Mercator coordinate
  ox, oy = latlonToMercator(origin_oxts[0],origin_oxts[1],scale)
  origin = np.array([ox,oy,0])
  
  oxts = []
  
  # for all oxts packets do
  for i in range(len(pose)):
    
    # if there is no data => no pose
    if not len(pose[i]):
      oxts.append([])
      continue
  
    # rotation and translation
    R = pose[i][0:3, 0:3]
    t = pose[i][0:3, 3]
  
    # unnormalize translation
    t = t+origin
  
    # translation vector
    lat, lon = mercatorToLatlon(t[0], t[1], scale)
    alt = t[2]
  
    # rotation matrix (OXTS RT3000 user manual, page 71/92)
    yaw = np.arctan2(R[1,0] , R[0,0])
    pitch = np.arctan2( - R[2,0] , np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = np.arctan2(R[2,1] , R[2,2])
  
    # rx = oxts{i}(4) # roll
    # ry = oxts{i}(5) # pitch
    # rz = oxts{i}(6) # heading 
    # Rx = [1 0 0 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]; # base => nav  (level oxts => rotated oxts)
    # Ry = [cos(ry) 0 sin(ry) 0 1 0; -sin(ry) 0 cos(ry)]; # base => nav  (level oxts => rotated oxts)
    # Rz = [cos(rz) -sin(rz) 0 sin(rz) cos(rz) 0; 0 0 1]; # base => nav  (level oxts => rotated oxts)
    # R  = Rz*Ry*Rx
        
    # add oxts 
    oxts.append([lat, lon, alt, roll, pitch, yaw])
  
  if single_value:
    oxts = oxts[0]
  
  return oxts
