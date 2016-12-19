function [frames] = get_video_frames(filename, requiredCnt)
    v = VideoReader(filename);    
    if ~exist('requiredCnt', 'var') || requiredCnt < 0
        requiredCnt = v.NumberOfFrames;
    else
        
    for frameCnt = 1:requiredCnt
    	if ~hasFrame(v)
        	break;
        end
        
        frame = readFrame(v);

        % Image even * even size
        %frame = modcrop(frame, 2);
        frames(:, :, :, frameCnt) = frame;
    end 
end