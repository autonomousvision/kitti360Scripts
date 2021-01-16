% Test script for loading Oxts data and convert to Mercator coordinate

% load poses
seq_id = 0;
kitti360_path = getenv('KITTI360_DATASET');
pose_file = sprintf('%s/data_poses/2013_05_28_drive_%04d_sync/poses.txt', kitti360_path, seq_id);
if ~exist(pose_file, 'file')
    error(fprintf('%s does not exist! \nPlease specify KITTI360_DATASET in your system path.\nPlease check if you have downloaded system poses (data_poses.zip) and unzipped them under KITTI360_DATASET' , pose_file));
end
[ts, poses] = loadPoses(pose_file);
fprintf('Loaded pose file %s\n', pose_file);

% convert coordinate system from
%   x=forward, y=left, z=up
% to
%   x=forward, y=right, z=down 
poses = postprocessPoses(poses);

% convert to lat/lon coordinate
oxts = convertPoseToOxts(poses);

% write to file
output_dir = 'output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
oxts = cell2mat(oxts);
oxts = reshape(oxts,6,[])';
output_file = sprintf('%s/2013_05_28_drive_%04d_sync_pose2oxts.txt', output_dir, seq_id);
dlmwrite(output_file, oxts, 'delimiter',' ','precision',6);
fprintf('Output written to %s\n', output_file);
