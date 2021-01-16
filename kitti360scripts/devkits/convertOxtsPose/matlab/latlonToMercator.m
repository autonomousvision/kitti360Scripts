function [mx,my] = latlonToMercator(lat,lon,scale)
% converts lat/lon coordinates to mercator coordinates using mercator scale

er = 6378137; % average earth radius at the equator
mx = scale * lon * pi * er / 180;
my = scale * er * log( tan((90+lat) * pi / 360) );
