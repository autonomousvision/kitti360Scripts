import numpy as np
from utils import latToScale, latlonToMercator

def convertOxtsToPose(oxts):
  ''' converts a list of oxts measurements into metric poses,
  starting at (0,0,0) meters, OXTS coordinates are defined as
  x = forward, y = right, z = down (see OXTS RT3000 user manual)
  afterwards, pose{i} contains the transformation which takes a
  3D point in the i'th frame and projects it into the oxts
  coordinates of the first frame. '''

  single_value = not isinstance(oxts, list)
  if single_value:
    oxts = [oxts]
  
  # origin in OXTS coordinate
  origin_oxts = [48.9843445, 8.4295857] # lake in Karlsruhe
  
  # compute scale from lat value of the origin
  scale = latToScale(origin_oxts[0])
  
  # origin in Mercator coordinate
  ox,oy = latlonToMercator(origin_oxts[0],origin_oxts[1],scale)
  origin = np.array([ox, oy, 0])
  
  pose     = []
  
  # for all oxts packets do
  for i in range(len(oxts)):
    
    # if there is no data => no pose
    if not len(oxts[i]):
      pose.append([])
      continue
  
    # translation vector
    tx, ty = latlonToMercator(oxts[i][0],oxts[i][1],scale)
    t = np.array([tx, ty, oxts[i][2]])
  
    # rotation matrix (OXTS RT3000 user manual, page 71/92)
    rx = oxts[i][3] # roll
    ry = oxts[i][4] # pitch
    rz = oxts[i][5] # heading 
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]]) # base => nav  (level oxts => rotated oxts)
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]]) # base => nav  (level oxts => rotated oxts)
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]]) # base => nav  (level oxts => rotated oxts)
    R  = np.matmul(np.matmul(Rz, Ry), Rx)
    
    # normalize translation
    t = t-origin
        
    # add pose
    pose.append(np.vstack((np.hstack((R,t.reshape(3,1))),np.array([0,0,0,1]))))
  
  if single_value:
    pose = pose[0]
  
  return pose
