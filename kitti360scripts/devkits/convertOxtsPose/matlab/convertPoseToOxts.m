function oxts = convertPoseToOxts(pose)
% converts a list of metric poses into oxts measurements,
% starting at (0,0,0) meters, OXTS coordinates are defined as
% x = forward, y = right, z = down (see OXTS RT3000 user manual)
% afterwards, pose{i} contains the transformation which takes a
% 3D point in the i'th frame and projects it into the oxts
% coordinates of the first frame.

single_value = ~iscell(pose);
if single_value
  pose_ = pose;
  clear pose;
  pose{1} = pose_;
end

% origin in OXTS coordinate
origin_oxts = [48.9843445, 8.4295857]; % lake in Karlsruhe

% compute scale from lat value of the origin
scale = latToScale(origin_oxts(1));

% origin in Mercator coordinate
[origin(1,1) origin(2,1)] = latlonToMercator(origin_oxts(1),origin_oxts(2),scale);
origin(3,1) = 0;

oxts     = [];

% for all oxts packets do
for i=1:length(pose)
  
  % if there is no data => no pose
  if isempty(pose{i})
    oxts{i} = [];
    continue;
  end

  % rotation and translation
  R = pose{i}(1:3, 1:3);
  t = pose{i}(1:3, 4);

  % unnormalize translation
  t = t+origin;

  % translation vector
  [lat, lon] = mercatorToLatlon(t(1), t(2), scale);
  alt = t(3);

  % rotation matrix (OXTS RT3000 user manual, page 71/92)
  yaw = atan2(R(2,1) , R(1,1));
  pitch = atan2( - R(3,1) , sqrt(R(3,2)^2 + R(3,3)^2));
  roll = atan2(R(3,2) , R(3,3));

  % rx = oxts{i}(4); % roll
  % ry = oxts{i}(5); % pitch
  % rz = oxts{i}(6); % heading 
  % Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]; % base => nav  (level oxts => rotated oxts)
  % Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]; % base => nav  (level oxts => rotated oxts)
  % Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]; % base => nav  (level oxts => rotated oxts)
  % R  = Rz*Ry*Rx;
      
  % add oxts 
  oxts{i} = [lat, lon, alt, roll, pitch, yaw];
  
end

if single_value
  oxts = oxts{1};
end
