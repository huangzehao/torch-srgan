require 'torch'
require 'image'
local M = {}

function M.crop_mod4(src)
   local w = src:size(3) % 4
   local h = src:size(2) % 4
   return M.crop(src, 0, 0, src:size(3) - w, src:size(2) - h)
end

function M.crop(src, w1, h1, w2, h2)
   local dest
   if src:dim() == 3 then
      dest = src[{{}, { h1 + 1, h2 }, { w1 + 1, w2 }}]:clone()
   else -- dim == 2
      dest = src[{{ h1 + 1, h2 }, { w1 + 1, w2 }}]:clone()
   end
   return dest
end

function M.random_crop(y, dst_size)
	local bs = y:size(1)
	local c = y:size(2)
	local w = y:size(4)
	local h = y:size(3)
	local dst, hi, wi
	local crop_y = torch.FloatTensor(bs, c, dst_size, dst_size)
	if w < h then
		assert(w >= dst_size, 'size of input image is smaller than crop size')
	else
		assert(h >= dst_size, 'size of input image is smaller than crop size')
	end
	for i = 1, bs do
		hi = torch.random(0, h - dst_size)
		wi = torch.random(0, w - dst_size)
		crop_y[i] = M.crop(y[i], wi, hi, wi + dst_size, hi + dst_size)
	end
	return crop_y
end 

function M.random_flip(x)
	local bs = y:size(1)
	local flip_mask = torch.randperm(bs):le(bs/2)
	for i = 1, bs do
		if flip_mask[i] == 1 then
    		image.hflip(y[i],y[i])
    	end
    end
    return y
end

function M.random_flip_2(x,y)
	local bs = y:size(1)
	local flip_mask = torch.randperm(bs):le(bs/2)
	for i = 1, bs do
		if flip_mask[i] == 1 then
    		image.hflip(y[i],y[i])
    		image.hflip(x[i],x[i])
    	end
    end
    return x, y
end
return M
