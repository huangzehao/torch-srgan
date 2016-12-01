require 'torch'
require 'optim'
require 'image'

local utils = require 'super_resolution.utils'
local gm = require 'graphicsmagick'
local cmd = torch.CmdLine()
-- Generic options
cmd:option('-img','./imgs/comic_input.bmp')
cmd:option('-output','output.bmp')
-- Super-resolution options
cmd:option('-use_tanh', false)
-- Checkpointing
cmd:option('-model', './models/SRResNet_MSE_100.t7')
-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 0)
cmd:option('-backend', 'cuda')

function main()
	local opt = cmd:parse(arg)

	-- Figure out the backend
	local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)

	-- Build the model
	local model = nil
	print('Loading model ...')
	model = torch.load(opt.model):type(dtype)

	if use_cudnn then cudnn.convert(model, cudnn) end

	model:evaluate()
	local img = gm.Image(opt.img):colorspace('RGB')
	local input = img:toTensor('float','RGB','DHW')
	input = torch.reshape(input,1,input:size(1),input:size(2),input:size(3))
	if opt.use_tanh then
		input = input:mul(2.0):add(-1.0)
	end
	local output = model:forward(input:type(dtype))
	local image = gm.Image(output[1]:float(),'RGB','DHW')
	image:save(opt.output)
end

main()


