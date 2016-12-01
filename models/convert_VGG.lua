require 'loadcaffe'

model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn')

-- move fc layers and last maxpooling layer
for _ = 1,10 do
	model:remove()
end

print(model)
torch.save('VGG19.t7', model)