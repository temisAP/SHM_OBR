%function [slices] = load_slices(dir_to)

    dir_to = '/mnt/sda/0_Andres/1_Universidad/SHM/98_data/0_Datasets/test_1/2_SLICES/mat';

    % Look subfolders
    dirinfo = dir(maindir);
    dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
    dirinfo(1:2) = [];
    
    % Look folders inside subfolders
    for i = 1:length(dirinfo)
      subdirinfo = dir([dirinfo(i).folder '/' dirinfo(i).name]);
      subdirinfo(~[subdirinfo.isdir]) = [];  %remove non-directories
      subdirinfo(1:2) = [];
      % Find .mat files
      for j = 1:length(subdirinfo)
          thisdir = [subdirinfo(j).folder '/' subdirinfo(j).name '/'];
          subsubdir = dir(fullfile(thisdir, '*.mat'));
          if i == 1 && j == 1 
            files = subsubdir;
          else
              files = cat(1, files, subsubdir);
          end
      end
    end

%end