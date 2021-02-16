function [ts, poses] = loadPoses (pos_file)

data = dlmread(pos_file);
ts = data(:,1);
pose = data(:,2:end);

poses = [];
for i = 1:size(pose, 1)
    poses{i} = [reshape(pose(i,:), [4, 3])'; 0 0 0 1];
end
