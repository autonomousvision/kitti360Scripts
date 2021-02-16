function poses = postprocessPoses (poses_in)

R = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1];

poses  = [];

for i = 1:length(poses_in)
    % if there is no data => no pose
    if isempty(poses_in{i})
      pose{i} = [];
      continue;
    end

    P = poses_in{i};
    poses{i} = (R * P')';
    %H = R * [H_; 0 0 0 1];
    %H = H(1:3,:)';
    %poses(p,:) = H(:);
end
