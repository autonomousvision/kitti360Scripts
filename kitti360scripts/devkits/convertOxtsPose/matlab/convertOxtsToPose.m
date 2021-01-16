function pose = convertOxtsToPose(oxts)
% converts a list of oxts measurements into metric poses,
% starting at (0,0,0) meters, OXTS coordinates are defined as
% x = forward, y = right, z = down (see OXTS RT3000 user manual)
% afterwards, pose{i} contains the transformation which takes a
% 3D point in the i'th frame and projects it into the oxts
% coordinates of the first frame.

single_value = ~iscell(oxts);
if single_value
  oxts_ = oxts;
  clear oxts;
  oxts{1} = oxts_;
end

% origin in OXTS coordinate
origin_oxts = [48.9843445, 8.4295857]; % lake in Karlsruhe

% compute scale from lat value of the origin
scale = latToScale(origin_oxts(1));

% origin in Mercator coordinate
[origin(1,1) origin(2,1)] = latlonToMercator(origin_oxts(1),origin_oxts(2),scale);
origin(3,1) = 0;

pose     = [];

% for all oxts packets do
for i=1:length(oxts)
  
  % if there is no data => no pose
  if isempty(oxts{i})
    pose{i} = [];
    continue;
  end

  % translation vector
  [t(1,1) t(2,1)] = latlonToMercator(oxts{i}(1),oxts{i}(2),scale);
  t(3,1) = oxts{i}(3);

  % rotation matrix (OXTS RT3000 user manual, page 71/92)
  rx = oxts{i}(4); % roll
  ry = oxts{i}(5); % pitch
  rz = oxts{i}(6); % heading 
  Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]; % base => nav  (level oxts => rotated oxts)
  Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]; % base => nav  (level oxts => rotated oxts)
  Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]; % base => nav  (level oxts => rotated oxts)
  R  = Rz*Ry*Rx;
  
  % normalize translation
  t = t-origin;
      
  % add pose
  pose{i} = [R t;0 0 0 1];
  
end

if single_value
  pose = pose{1};
end
