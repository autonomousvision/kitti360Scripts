function [lat,lon] = mercatorToLatlon(mx,my,scale)
% converts mercator coordinates using mercator scale to lat/lon coordinates

er = 6378137; % average earth radius at the equator
lon = mx * 180 / (scale * pi * er); 
lat = 360 / pi * atan(exp(my / (scale * er))) - 90;
