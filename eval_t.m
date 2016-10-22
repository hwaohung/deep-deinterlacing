function [indexes] = eval_t(patches, t, greater)
    if greater
        indexes = patches(4, 4, :, :) >= t;
    else
        indexes = patches(4, 4, :, :) < t;
    end
end