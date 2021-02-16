function [oxts,ts] = loadOxtsData(oxts_dir,frames)
% reads GPS/IMU data from files to memory. requires base directory
% (=sequence directory as parameter). if frames is not specified, loads all frames.

ts = [];

if nargin==1

  ts = loadTimestamps(oxts_dir);
  oxts  = [];
  for i=1:length(ts)
    oxts{i} = [];
    if ~isempty(ts{i})
      try
        oxts{i} = dlmread([oxts_dir '/data/' num2str(i-1,'%010d') '.txt']);
      catch e
        oxts{i} = [];
      end
    end
  end
   
else

  if length(frames)>1
    k = 1;
    oxts = [];
    for i=1:length(frames)
      try
        file_name = [oxts_dir '/data/' num2str(frames(i)-1,'%010d') '.txt'];
        oxts{k} = dlmread(file_name);
      catch e
        oxts{k} = [];
      end
      k=k+1;
    end
    
  % no cellarray for single value
  else
    file_name = [oxts_dir '/data/' num2str(frames(1)-1,'%010d') '.txt'];
    try
      oxts = dlmread(file_name);
    catch e
      oxts = [];
    end
  end

end
