----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 2-class problem: faces!
local noutputs = 2

-- input dimensions: faces!
local nfeats = 3
local width = 256
local height = 256

-- hidden units, filter sizes (for ConvNet only):
local nstates = {16,32}
local filtsize = {5, 7}
local poolsize = 4

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')

local CNN = nn.Sequential()

-- stage 1: conv+max
CNN:add(nn.SpatialConvolutionMM(3, 16, 5, 5, 1, 1, 0, 0))
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(2,2,2,2))

-- stage 2: conv+max
CNN:add(nn.SpatialConvolutionMM(16, 16, 12, 12, 2, 2, 0, 0))
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(3,3,1,1))
CNN:add(nn.SpatialConvolutionMM(16, 32, 4, 4, 4, 4, 0, 0))
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(2,2,2,2))

local classifier = nn.Sequential()
-- stage 3: linear
classifier:add(nn.Reshape(7*7*32))
classifier:add(nn.Linear(7*7*32, 2))

-- stage 4 : log probabilities
classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end

local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

