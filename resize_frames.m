function [resizeds1, resizeds2] = resize_frames(frames)
    for i = 1:size(frames, 3)
        frame = frames(:, :, i);
        if mod(i, 2) == 1
            resizeds1(:, :, (i+1)/2)  = frame(1:2:end, 1:2:end);
        else
            resizeds2(:, :, i/2)  = frame(2:2:end, 2:2:end);
        end
    end
end