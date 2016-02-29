require 'torch'
require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'
require 'math'
require 'pl'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 32)         batch size
   -r,--learningRate          (default 10)           learning rate
   --learningRateDecay        (default 1e-7)        learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --observation_step         (default 25)          observation step
   --model                    (default vgg_bn_drop) model name
   --max_observations         (default 100000)      maximum number of observations
   --sgdSteps                 (default 300)          sgd steps before sgdsvd
   --observerSteps            (default 40)          steps between test observations
   --backend                  (default nn)          backend
   --visualize                (default true)        visualize weigths
]]

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = true

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  sgdSteps = opt.sgdSteps,
  learningRate = opt.learningRate
}

function display(input)
  iter = iter or 0
  require 'image'
  win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
  if iter % 10 == 0 then
    win_w1 = image.display{
      image=model:get(3):get(1).weight, zoom=4, nrow=10,
      min=-1, max=1,
      win=win_w1, legend='stage 1: weights', padding=1
      }
    win_w2 = image.display{
      image=model:get(3):get(54):get(6).weight, zoom=4, nrow=10,
      min=-1, max=1,
      win=win_w2, legend='stage 2: weights', padding=1
      }
   end
   iter = iter + 1
end

function train(trainData)
  model:training()
  observation = observation or 1
  
  print(c.blue '==>'.." online observation # " .. observation .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(opt.observerSteps * opt.batchSize):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    --xlua.progress(t, #indices)

    local inputs = trainData.data:index(1,v)
    targets:copy(trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)
      
      -- visualize?
            if opt.visualize then
               display(inputs[1])
            end

      return f,gradParameters
    end
    optim.sgdsvd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  observation = observation + 1
end


function test(testData)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 32
  for i = 1, testData.data:size(1), bs do
    local outputs = model:forward(testData.data:narrow(1, i, bs))
    confusion:batchAdd(outputs, testData.labels:narrow(1, i, bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    currentTensorType = torch.Tensor():type()
    torch.setdefaulttensortype('torch.FloatTensor')
    testLogger:plot()
    torch.setdefaulttensortype(currentTensorType)

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model
  if observation % 30 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion:zero()
end


for i = 1, opt.max_observations do
  local trainIndices = torch.randperm(provider.trainData.data:size(1)):narrow(1, 1, opt.batchSize * opt.observerSteps)
  local trainData = {
     data = torch.Tensor(provider.trainData.data:size(1), 3072),
     labels = torch.Tensor(provider.trainData.data:size(1)),
     size = function() return trsize end
  }
  trainData.data = provider.trainData.data:index(1, trainIndices:long())
  trainData.labels = provider.trainData.labels:index(1, trainIndices:long())
  
  local testIndices = torch.randperm(provider.testData.data:size(1)):narrow(1, 1, opt.batchSize * opt.observerSteps)
  local testData = {
     data = torch.Tensor(provider.testData.data:size(1), 3072),
     labels = torch.Tensor(provider.testData.data:size(1)),
     size = function() return trsize end
  }
  testData.data = provider.testData.data:index(1, testIndices:long())
  testData.labels = provider.testData.labels:index(1, testIndices:long())
  
  train(trainData)
  test(testData)
end
