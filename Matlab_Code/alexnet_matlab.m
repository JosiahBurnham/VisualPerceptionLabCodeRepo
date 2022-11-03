%----------------------------------------------------------
% File   : alexnet_matlab.m 
% Author : J.Burnham
% Date   : 03/14/2022
% Purpose: To create a script that will allow the caller to select a custom 
%          Deep learning model that is a trained alexnet with only specified
%          layers in it
%----------------------------------------------------------
classdef alexnet_matlab

    %------------------------------------------------------
    % Class attributes
    %------------------------------------------------------
    properties
        layer_to_chop;
        num_outputs;
    end

    methods
        %--------------------------------------------------
        % Constructor(s)
        %--------------------------------------------------
        function obj = alexnet_matlab(layer_val, outputs)
            obj.layer_to_chop = layer_val;
            obj.num_outputs = outputs;
        end

        %--------------------------------------------------
        % Class Methods
        %--------------------------------------------------

        %--------------------------------------------------
        % this function returns a pretrained alexnet model chopped of where
        % the call specifies
        % Returns:
        %   a:[trained model] - a alexnet model chopped off at a specified
        %   level
        %--------------------------------------------------
        function choppedNet = call(obj)
            layers = [];
            net = alexnet;
            % input -> first max pool
            if obj.layer_to_chop >= 1
                layers = [... % first block of layers in alexnet
                           layers,...
                           net.Layers(1),... % input
                           RGBLayer,...      % custom convert to RGB layer
                           net.Layers(2),... % convolution
                           net.Layers(3),... % Relu
                           net.Layers(4),... % cross channel normalization
                           net.Layers(5)     % Max pool
                         ];
            end

            % second convolution -> second max pool
            if obj.layer_to_chop >=6
                layers = [... % second block of layers in alexnet
                           layers,...       
                           net.Layers(6),... % Group Convolution
                           net.Layers(7),... % Relu
                           net.Layers(8),... % Cross Channel Normalization
                           net.Layers(9)     % Max pool
                         ];
            end

            % third convolution -> thrid max pool
            if obj.layer_to_chop >=10
                layers = [... % third block of layers in alexnet
                           layers,...
                           net.Layers(10),... % Convolution
                           net.Layers(11),... % Relu
                           net.Layers(12),... % Grouped Convolution
                           net.Layers(13),... % Relu
                           net.Layers(14),... % Grouped Convolution
                           net.Layers(15),... % Relu
                           net.Layers(16)     % Max pool
                         ];
            end

            % first fully connected layer
            if obj.layer_to_chop >= 17
                layers = [... % fourth block of layers in alexnet
                           layers,...
                           net.Layers(17),... % Dense
                           net.Layers(18),... % Relu
                           net.Layers(19)     % Dropout
                         ];
            end

            % second fully connected layer
            if obj.layer_to_chop >= 20
                layers = [... % fifth block of layers in alexnet
                           layers,...
                           net.Layers(20),... % Dense
                           net.Layers(21),... % Relu
                           net.Layers(22)     % Dropout
                         ];
            end

            % add an output layer to the network
            layers = [...
                        layers,...
                        fullyConnectedLayer(obj.num_outputs),...
                        softmaxLayer,...
                        classificationLayer
                     ];

            lgraph = layerGraph(layers);
            figure
            plot(lgraph)
            choppedNet = lgraph;
        end % end call()
    end % end methods
end % end alexnet_matlab