%----------------------------------------------------------
% File   : RGBLayer.m 
% Author : J.Burnham
% Date   : 03/16/2022
% Purpose: A custom deep learning layer that will allow a greyscale iamge
% to be converted to RGB by creating a new image with three copies of the
% orginal greyscale image stacked on top of each other
%----------------------------------------------------------
classdef RGBLayer < nnet.layer.Layer
    %------------------------------------------------------
    % Class Methods
    %------------------------------------------------------
    methods
        %--------------------------------------------------
        % This custom layer convets an greyscale image to RGB
        % Parameters:
           
        % Returns:
        %   Z:[RGB image]
        %--------------------------------------------------
        function Z = predict(layer, X)

            RGB = cat(4, X, X, X);
            Z = RGB;
        end % end predict
    end % end methods
end % end RGBLayer