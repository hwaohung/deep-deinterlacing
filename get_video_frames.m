% Gray only
function [frames] = get_video_frames(filename, requiredCnt)
    v = VideoReader(filename);    
    if ~exist('requiredCnt', 'var')
        % TODO: Not sure
        requiredCnt = v.FrameRate;
    end
    
    for frameCnt = 1:requiredCnt
    	if ~hasFrame(v)
        	break;
        end
        
        frame = readFrame(v);      
        if size(frame, 3) > 1
            frame = rgb2gray(frame);
        end
        
        % Image even * even size(why)
        frame = modcrop(frame, 2);
        frames(:, :, frameCnt) = frame;
    end
end