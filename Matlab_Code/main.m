% layer = RGBLayer;
% validInputSize = uint8(zeros(100,40,40,1));
% layer_output = layer.predict(validInputSize);
% size(layer_output)
%checkLayer(layer, validInputSize)

net = alexnet;

conv1 = net.Layers(2).Weights;
conv2 = net.Layers(6).Weights(:,:,:,:,1);
conv3 = net.Layers(6).Weights(:,:,:,:,2);
conv4 = net.Layers(10).Weights;
conv5 = net.Layers(12).Weights(:,:,:,:,1);
conv6 = net.Layers(12).Weights(:,:,:,:,2);
conv7 = net.Layers(14).Weights(:,:,:,:,1);
conv8 = net.Layers(14).Weights(:,:,:,:,2);
fc1 = net.Layers(17).Weights;
fc2 = net.Layers(20).Weights;

