----------------------------------------------------------------------
-- This script demonstrates how to load the Face Detector 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet, Eugenio Culurciello
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}


print(sys.COLORS.red ..  '==> loading dataset')

-- We load the dataset from disk

local imagesAll = torch.Tensor(7319,3,256,256)
local labelsAll = torch.Tensor(7319)

-- classes: GLOBAL var!
classes = {'bg','bumper'}

-- load background:
for f=0,4881 do
  k=f+1
  print(k)
  imagesAll[f+1] = image.load('neg/neg ('..k..').png',3) 
  labelsAll[f+1] = 1 -- 1 = first tool
end
-- load bompor:
for f=4882,7318 do
  k=f-4881
  print(k)
  imagesAll[f+1] = image.load('pos/pos ('..k..').png',3) 
  labelsAll[f+1] = 2 -- 1 = second tool
end


-- shuffle dataset: get shuffled indices in this variable:
local labelsShuffle = torch.randperm((#labelsAll)[1])

local portionTrain = 0.8 -- 80% is train data, rest is test data
local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = labelsShuffle:size(1) - trsize

-- create train set:
trainData = {
   data = torch.Tensor(trsize, 3, 256, 256),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
--create test set:
testData = {
      data = torch.Tensor(tesize, 3, 256, 256),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }



for i=1,trsize do
   trainData.data[i] = imagesAll[labelsShuffle[i]]:clone()
   trainData.labels[i] = labelsAll[labelsShuffle[i]]
end
for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]]:clone()
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> preprocessing data')

-- Name channels for convenience
local channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


print(sys.COLORS.red ..  '==> verify statistics')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   -- local first256Samples_y = trainData.data[{ {1,256},1 }]
   -- image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   -- local first256Samples_y = testData.data[{ {1,256},1 }]
   -- image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}


