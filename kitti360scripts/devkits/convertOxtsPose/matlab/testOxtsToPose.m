% Test script for loading Oxts data and convert to Mercator coordinate

% load Oxts data
seq_id = 0;
kitti360_path = getenv('KITTI360_DATASET');
oxts_dir = sprintf('%s/data_poses/2013_05_28_drive_%04d_sync/oxts', kitti360_path, seq_id);
if ~exist(oxts_dir, 'dir')
    error(fprintf('%s does not exist! \nPlease specify KITTI360_DATASET in your system path.\nPlease check if you have downloaded system poses (data_poses.zip) and unzipped them under KITTI360_DATASET' , oxts_dir));
end
[oxts,ts] = loadOxtsData(oxts_dir);
fprintf('Loaded oxts data from %s\n', oxts_dir);

% convert to Mercator coordinate
poses = convertOxtsToPose(oxts);

% convert coordinate system from
%   x=forward, y=right, z=down 
% to
%   x=forward, y=left, z=up
poses = postprocessPoses(poses);

% write to file
output_dir = 'output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
poses = cell2mat(poses);
poses = permute(reshape(poses,4,4,[]), [2,1,3]);
poses = reshape(poses,16,[])';
output_file = sprintf('%s/2013_05_28_drive_%04d_sync_oxts2pose.txt', output_dir, seq_id);
dlmwrite(output_file, poses, 'delimiter',' ','precision',6);
fprintf('Output written to %s\n', output_file);
