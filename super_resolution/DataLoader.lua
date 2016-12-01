require 'torch'
require 'hdf5'
local dataprocess = require 'super_resolution.dataprocess'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	assert(opt.h5_file, 'Must provide h5_file')
	assert(opt.batch_size, 'Must provide batch size')

	self.h5_file = hdf5.open(opt.h5_file,'r')
	self.batch_sizes = {
		train = opt.batch_size,
		val = 1,
	}
	self.split_idxs = {
		train = 1,
		val = 1,
	}

	self.image_paths = {
		train = 'train',
		val = 'val',
	}

	local train_label_size = self.h5_file:read(string.format('%s_label',self.image_paths.train)):dataspaceSize()
	local train_data_size = self.h5_file:read(string.format('%s_data',self.image_paths.train)):dataspaceSize()
	local val_label_size = self.h5_file:read(string.format('%s_label',self.image_paths.val)):dataspaceSize()
	local val_data_size = self.h5_file:read(string.format('%s_data',self.image_paths.val)):dataspaceSize()

	self.split_sizes = {
		train = train_label_size[1],
		val = val_label_size[1],
	}

	self.num_channels = train_label_size[2]
	self.label_image_height = train_label_size[3]
	self.label_image_width = train_label_size[4]
	self.data_image_height = train_data_size[3]
	self.data_image_width = train_data_size[4]

	self.random_flip = opt.random_flip

	self.num_minibatches = {}
	for k, v in pairs(self.split_sizes) do
		self.num_minibatches[k] = math.floor(v / self.batch_sizes[k])
	end
end

function DataLoader:reset(split)
	self.split_idxs[split] = 1
end

function DataLoader:getBatch(split)
	local path = self.image_paths[split]
	local start_idx = self.split_idxs[split]
	local end_idx = math.min(start_idx + self.batch_sizes[split] - 1, self.split_sizes[split])

	local label_images = self.h5_file:read(string.format('%s_label',path)):partial(
							{start_idx, end_idx},
							{1, self.num_channels},
							{1,self.label_image_height},
							{1, self.label_image_width}):float():div(255)
	local data_images = self.h5_file:read(string.format('%s_data',path)):partial(
							{start_idx, end_idx},
							{1, self.num_channels},
							{1, self.data_image_height},
							{1, self.data_image_width}):float():div(255)
	if split == 'train' and self.random_flip then
		data_images, label_images = dataprocess.random_flip_2(data_images,label_images)
	end
	self.split_idxs[split] = end_idx + 1
	if self.split_idxs[split] > self.split_sizes[split] then
		self.split_idxs[split] = 1
	end
	return data_images, label_images
end