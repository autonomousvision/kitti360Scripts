import numpy as np

er = 6378137. # average earth radius at the equator

def latlonToMercator(lat,lon,scale):
  ''' converts lat/lon coordinates to mercator coordinates using mercator scale '''

  mx = scale * lon * np.pi * er / 180;
  my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) );
  return mx,my

def mercatorToLatlon(mx,my,scale):
  ''' converts mercator coordinates using mercator scale to lat/lon coordinates '''

  lon = mx * 180. / (scale * np.pi * er) 
  lat = 360. / np.pi * np.arctan(np.exp(my / (scale * er))) - 90.
  return lat, lon

def latToScale(lat):
  ''' compute mercator scale from latitude '''
  scale = np.cos(lat * np.pi / 180.0);
  return scale

def postprocessPoses (poses_in):

  R = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
  
  poses  = []
  
  for i in range(len(poses_in)):
    # if there is no data => no pose
    if not len(poses_in[i]):
      pose.append([])
      continue
  
    P = poses_in[i]
    poses.append( np.matmul(R, P.T).T )
  
  return poses
