function ts = loadTimestamps(ts_dir)

fid = fopen([ts_dir '/timestamps.txt']);
col = textscan(fid,'%s\n',-1,'delimiter',',');
ts = col{1};
fclose(fid);
