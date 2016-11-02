function [deinterlaced_frames] = deinterlace_video(frames, requiredCnt)
    tmp1 = frames(:, :, 1);
    tmp1(2:2:end, :) = 0;
    
    folder = '';
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
    
    throw('Not found video');
end