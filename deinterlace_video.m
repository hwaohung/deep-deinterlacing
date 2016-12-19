function [deinterlaced_frames] = deinterlace_video(frames, requiredCnt)
    tmp1 = frames(:, :, :, 1);
    
    folder = 'v_test';
    deinterlaced_frames = find_same(tmp1, folder, requiredCnt);
    
    if size(frames, 1) == size(deinterlaced_frames, 1)
        return;
    end
    
    folder = 'v_train';
    deinterlaced_frames = find_same(tmp1, folder, requiredCnt);
    
    if size(frames, 1) == size(deinterlaced_frames, 1)
        return;
    end
    
    throw(MException('MYFUN:BadIndex', 'Not found video'));
end

function [deinterlaced_frames] = find_same(tmp1, folder, requiredCnt)
    deinterlaced_frames = 0;
    filepaths = dir(fullfile(folder, '*.avi'));
    for i = 1:length(filepaths)
        tmp2 = get_video_frames(fullfile(folder, filepaths(i).name), 1);
        
        diff = abs(tmp1(1:2:end, :, :, :) - tmp2(1:2:end, :, :, :));
        diff = sum(diff(:))
        
        if diff == 0
            deinterlaced_frames = get_video_frames(fullfile(folder, filepaths(i).name), requiredCnt);
        end
    end
end