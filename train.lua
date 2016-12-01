require 'torch'
require 'optim'
require 'image'
require 'xlua'
require 'super_resolution.DataLoader'

local utils = require 'super_resolution.utils'
local models = require 'super_resolution.models'
local gm = require 'graphicsmagick'
local c = require 'trepl.colorize'
local cmd = torch.CmdLine()
-- Generic options
cmd:option('-h5_file','/data/imagenet-val-192.h5')
cmd:option('-val_img','./imgs/comic_input.bmp')
cmd:option('-val_output','./val/')
cmd:option('-residual_blocks', 15)
cmd:option('-deconvolution_type','sub_pixel','sub_pixel|fullconvolution')
cmd:option('-debug', false)
-- Super-resolution options
cmd:option('-loss', 'pixel', 'pixel|percep')
cmd:option('-percep_layer', 'conv2_2', 'conv2_2|conv5_4')
cmd:option('-percep_model', './models/VGG19.t7')
cmd:option('-use_tanh', false)
-- Optimization
cmd:option('-num_epoch', 100)
cmd:option('-batch_size', 16)
cmd:option('-learning_rate', 1e-3)
cmd:option('-beta1', 0.9)
cmd:option('-weight_decay', 0)
cmd:option('-random_flip', true)
-- Checkpointing
cmd:option('-resume_from_checkpoint', '')
cmd:option('-resume_epoch',0)
cmd:option('-checkpoint_name', './checkpoint/checkpoint')
cmd:option('-checkpoint_every', 1)
-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda')

function main()
	local opt = cmd:parse(arg)

	-- Figure out the backend
	local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)

	-- Build the model
	local model = nil
	if opt.resume_from_checkpoint ~= '' then
		print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
		model = torch.load(opt.resume_from_checkpoint):type(dtype)
	else
		print('Initializing model from scratch')
		model = models.build_model(opt):type(dtype)
	end
	if use_cudnn then cudnn.convert(model, cudnn) end
	model:training()
	--print(model)

	local criterion = nn.MSECriterion():type(dtype)

	local loader = DataLoader(opt)
	local params, grad_params = model:getParameters()

	-- Load percep model
	local percep_model = nil
	local params_percep, grad_params_percep = nil
	if opt.loss == 'percep' then
		print('Training with perceptual loss of layer ' .. opt.percep_layer)
		print('Loading VGG19 model')
		percep_model = torch.load(opt.percep_model)
		if opt.percep_layer == 'conv2_2' then
			for _ = 1,27 do
				percep_model:remove()
			end
		end
		percep_model:type(dtype)
		if use_cudnn then cudnn.convert(percep_model, cudnn) end
		percep_model:evaluate()
		params_percep, grad_params_percep = percep_model:getParameters()
		print(percep_model)
	end

	local function f(x)
		assert(x == params)
		grad_params:zero()

		-- Load data and label
		local x, y = loader:getBatch('train')
		if opt.use_tanh then
			x = x:mul(2.0):add(-1.0)
		end

		x, y = x:type(dtype), y:type(dtype)
		-- Run model forward
		local out = model:forward(x)
		local grad_out = nil

		-- Compute loss and loss gradient
		local loss = 0
		if opt.loss == 'pixel' then
			loss = criterion:forward(out, y)
			grad_out = criterion:backward(out, y)
		elseif opt.loss == 'percep' then
			grad_params_percep:zero()
			local input_real_percep = utils.vgg_preprocess(y)
			local input_sr_percep = utils.vgg_preprocess(out)
			local output_real_percep = percep_model:forward(input_real_percep):clone()
			local output_sr_percep = percep_model:forward(input_sr_percep)
			loss = criterion:forward(output_sr_percep, output_real_percep)
			local percep_grad_out = criterion:backward(output_sr_percep, output_real_percep)
			local percep_grad_in = percep_model:backward(input_sr_percep, percep_grad_out)
			grad_out = percep_grad_in:mul(255.0):index(2, torch.LongTensor{3,2,1})
		end
		model:backward(x, grad_out)
		return loss, grad_params
	end
	local optim_state = {learningRate=opt.learning_rate,
						beta1 = opt.beta1,
						weightDecay = opt.weight_decay,
						}
	local train_loss_history = {}
	local val_loss_history = {}
	-- Training
	for epoch = opt.resume_epoch + 1, opt.num_epoch do
		local tic = torch.tic()
		print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batch_size .. ' lr = ' .. optim_state.learningRate .. ']')
		local loss_epoch = 0
		for t = 1, loader.num_minibatches['train'] do
			local _, loss_batch = optim.adam(f, params, optim_state)
			loss_epoch = loss_epoch + loss_batch[1]
			if opt.debug then
				print(string.format('Epoch %d, Iteration %d / %d, loss = %f ', 
							epoch, t, loader.num_minibatches['train'], loss_batch[1]), optim_state.learningRate)
			else
				xlua.progress(t, loader.num_minibatches['train'])
			end
		end
		loss_epoch = loss_epoch / loader.num_minibatches['train']
		print(('Train loss: '..c.cyan'%.6f'..' \t time: %.2f s'):format(loss_epoch, torch.toc(tic)))
        table.insert(train_loss_history, loss_epoch)

		-- Testing
        if epoch % opt.checkpoint_every == 0 then
        	-- Check loss on the validation set
        	loader:reset('val')
			model:evaluate()
			local val_loss = 0
			print('Running on validation set')
			local val_batches = loader.num_minibatches['val']
			for j = 1, val_batches do
				local x, y = loader:getBatch('val')
				if opt.use_tanh then
					x = x:mul(2.0):add(-1.0)
				end
				x, y = x:type(dtype), y:type(dtype)
				local out = model:forward(x)
				local loss = 0
				if opt.loss == 'pixel' then
					loss = criterion:forward(out, y)
				elseif opt.loss == 'percep' then
					local input_real_percep = utils.vgg_preprocess(y)
					local input_sr_percep = utils.vgg_preprocess(out)
					local output_real_percep = percep_model:forward(input_real_percep):clone()
					local output_sr_percep = percep_model:forward(input_sr_percep)
					loss = criterion:forward(output_sr_percep, output_real_percep)
				end
				val_loss = val_loss + loss
			end
			val_loss = val_loss / val_batches
			print(('Val loss: '..c.cyan'%.6f'):format(val_loss))
			table.insert(val_loss_history, val_loss)
			-- Save log
        	local log = {opt = opt, 
						train_loss_history = train_loss_history,
						val_loss_history = val_loss_history,
						}
			local filename = string.format('%s.json',opt.checkpoint_name)
			paths.mkdir(paths.dirname(filename))
			utils.write_json(filename, log)

			-- Check performance on the val img
			local val_img = gm.Image(opt.val_img):colorspace('RGB')
			local input = val_img:toTensor('float','RGB','DHW')
			input = torch.reshape(input,1,input:size(1),input:size(2),input:size(3))
			if opt.use_tanh then
				input = input:mul(2.0):add(-1.0)
			end
			local output = model:forward(input:type(dtype))
			local image = gm.Image(output[1]:float(),'RGB','DHW')
			image:save(opt.val_output .. 'outputs_' .. epoch .. '.bmp')

			-- Save model
			model:clearState()
			if use_cudnn then 
				cudnn.convert(model, nn)
			end
			model:float()
			filename = string.format('%s_%d.t7',opt.checkpoint_name,epoch)
			torch.save(filename,model)
			model:type(dtype)
			if use_cudnn then
				cudnn.convert(model,cudnn)
			end
			params, grad_params = model:getParameters()
		end
		model:training()
	end

end

main()


