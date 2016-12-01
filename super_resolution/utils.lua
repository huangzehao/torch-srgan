require 'torch'
require 'nn'
local cjson = require 'cjson'
local M = {}

function M.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end


function M.write_json(path, j)
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end


local function check_input(img)
	assert(img:dim() == 4, 'img must be N x C x H x W')
	assert(img:size(2) == 3, 'img must have three channels')
end

function M.setup_gpu(gpu, backend, use_cudnn)
	local dtype = 'torch.FloatTenser'
	if gpu >= 0 then
		if backend == 'cuda' then
			require 'cutorch'
			require 'cunn'
			cutorch.setDevice(gpu + 1)
			dtype = 'torch.CudaTensor'
			if use_cudnn then
				require 'cudnn'
				cudnn.benchmark = true
			end
		end
	end
	return dtype, use_cudnn
end

function M.init_msra(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end
function M.init_BN(model)
	for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
		v.weight:fill(1)
		v.bias:zero()
	end
end
local vgg_mean = {103.939, 116.779, 123.68}

function M.vgg_preprocess(img)
	check_input(img)
	local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
	local perm = torch.LongTensor{3, 2, 1}
	return img:index(2, perm):mul(255):add(-1, mean)
end
return M
