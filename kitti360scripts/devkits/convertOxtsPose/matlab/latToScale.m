function scale = latToScale(lat)
% compute mercator scale from latitude

scale = cos(lat * pi / 180.0);
