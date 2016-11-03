require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
require 'cunn'
require 'image'
local cv = require 'cv'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
----------------------------------------------------------------------

-- model:
--torch.setdefaulttensortype('torch.FloatTensor')
local t
t=torch.load('results/model.net')
t = t:cuda()
local i=0
local j=0
loadType = cv.IMREAD_UNCHANGED
fullim = cv.imread{'2.jpg', loadType}
result = cv.imread{'result.jpg',loadType}
while i<64 do
	while j<29 do
		local crop = cv.getRectSubPix{image=fullim, patchSize={256,256}, center={i*16+128,j*16+128}}
		cv.imwrite{'test.jpg', crop}
		local img = image.load('test.jpg')
		local channels = {'y','u','v'}
		local mean = {}
		local std = {}
		for i,channel in ipairs(channels) do
			mean[i] = img[i]:mean()
			std[i] = img[i]:std()
			img[i]:add(-mean[i])
			img[i]:div(std[i])
		end
		img = img:cuda()
		local preds = t:forward(img)
		if preds[1]>preds[2] then
			cv.circle{result, {i*16+128, j*16+128}, 4, {0, 150, 0}, 7}
		elseif preds[2]>preds[1]/15 then
			print("bumper")
			cv.circle{result, {i*16+128, j*16+128}, 4, {150, 0, 0}, 7}
		else

		end
		cv.imwrite{'result.jpg', result}
		-- print(preds[1])
		-- print(preds[2])
	j=j+1
	end
j=0
i=i+1
end
	
-- while i<160 do
-- 	while j<112 do
-- 		local inputs = torch.Tensor(1,1,32,32)
-- 		local img = image.load('imageData/imi'..i..'j'..j..'.png')
-- 		local channels = {'y','u','v'}
-- 		local mean = {}
-- 		local std = {}
		
-- 		for i,channel in ipairs(channels) do
-- 		   mean[i] = img[1]:mean()
-- 		   std[i] = img[1]:std()
-- 		   img[1]:add(-mean[i])
-- 		   img[1]:div(std[i])
-- 		end

-- 		inputs[1]=img[1]
-- 		local preds = t:forward(inputs)
-- 		-- print(preds[1])
		
--         if preds[1][1]>preds[1][2] and preds[1][1]>preds[1][3] then
--         print('..i..,..j.. is a tool 1')
-- 		src=image.drawRect(src, i, j, i+2, j+2, {lineWidth = 2, color = {255, 0, 0}})
--         end
--         if preds[1][2]>preds[1][1] and preds[1][2]>preds[1][3] then
-- 		print('..i..,..j.. is a tool 2')
-- 		src=image.drawRect(src, i, j, i+2, j+2, {lineWidth = 2, color = {0, 255, 0}})
--         end
--         if preds[1][3]>preds[1][1] and preds[1][3]>preds[1][2] then
-- 		print(''..i..','..j..' is a background')
-- 		src=image.drawRect(src, i, j, i+2, j+2, {lineWidth = 2, color = {0, 0, 255}})
--         end
		
--     	j=j+2
-- 	end
-- 	i=i+2
-- 	j=0
-- end
-- local img = image.load('bompor.jpg',3)






