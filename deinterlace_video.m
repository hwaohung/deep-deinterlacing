function [deinterlaced_frames] = deinterlace_video(frames, requiredCnt)
    tmp1 = frames(:, :, 1);
    tmp1(2:2:end, :) = 0;
    
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
    
    throw('Not found video');
end

function [deinterlaced_frames] = find_same(tmp1, folder, requiredCnt)
    filepaths = dir(fullfile(folder, '*.avi'));
    for i = 1:length(filepaths)
        tmp2 = get_video_frames(fullfile(folder, filepaths(i).name), 1);
        tmp2(2:2:end, :) = 0;
        
        if ~same(tmp1, tmp2)
            continue;
        end
        
        score = tmp1 == tmp2;
        score = sum(score(:));
        
        if score == size(tmp1, 1) * size(tmp1, 2)
            deinterlaced_frames = get_video_frames(fullfile(folder, filepaths(i).name), requiredCnt);
            return;
        end
    end
    
    deinterlaced_frames = 0;
end