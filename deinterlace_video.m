function [deinterlaced_frames, prpo_frames] = deinterlace_video(frames, requiredCnt)
    tmp1 = frames(:, :, :, 1);
    
    folder = 'v_test';
    filename = find_same(tmp1, folder);
    
    prefix = 'prpo_';
    
    if filename ~= 0
        deinterlaced_frames = get_video_frames(fullfile(folder, filename), requiredCnt);
        prpo_frames = get_video_frames(fullfile(folder, [prefix filename]), requiredCnt);
        return;
    end
    
    folder = 'v_train';
    filename = find_same(tmp1, folder);
    
    if filename ~= 0
        deinterlaced_frames = get_video_frames(fullfile(folder, filename), requiredCnt);
        prpo_frames = get_video_frames(fullfile(folder, [prefix filename]), requiredCnt);
        return;
    end
    
    throw(MException('MYFUN:BadIndex', 'Not found video'));
end

function [filename] = find_same(tmp1, folder)
    filename = 0;
    filepaths = dir(fullfile(folder, '*.avi'));
    for i = 1:length(filepaths)
        tmp2 = get_video_frames(fullfile(folder, filepaths(i).name), 1);
        
        diff = abs(tmp1(1:2:end, :, :, :) - tmp2(1:2:end, :, :, :));
        diff = sum(diff(:));
        
        if diff == 0
            filename = filepaths(i).name;
            break;
        end
    end
end