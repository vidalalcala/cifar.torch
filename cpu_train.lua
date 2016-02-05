----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR10 
-- dataset, using SGD and SGDOLS
--
-- Jose V. Alcala-Burgos
-- vidal.alcala@gmail.com
----------------------------------------------------------------------
--

--DEBUG LINE
--require('mobdebug').start()
--


require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
local c = require 'trepl.colorize'
require 'math'

--Set parameters (used in DEBUG mode)
--myArg = {}
--myArg[0] = 'train.lua'
--myArg[1] = '-p'
--myArg[-2] = '-e'
--myArg[-1] = "require 'torch-env'"
--myArg[-3] = '/usr/local/bin/torch-qlua'
--_G["arg"] = myArg

-- my opt library
--require 'sgdolsInit'
--require 'svd'

-- luatrace
--local luatrace = require('luatrace.profile')
--luatrace.tron()

-- use CUDA
-- torch.setdefaulttensortype('torch.CudaTensor')

-- SGDOLS function
--function optim.sgdols(opfunc, x, state)
--   
--   -- (0) get/update state
--   local lr = 1.00
--   local gamma = 0.60
--   state.evalCounter = state.evalCounter or 0
--   local nevals = state.evalCounter
--   local p = state.numParameters
--   local P = state.P
--   local B = state.B
--   local G = state.G
--   local Gt = state.Gt
--   
--   -- (1) evaluate f(x) and df/dx
--   local fx,dfdx = opfunc(state.parametersSlow)
--   
--    -- start trace
--   --luatrace = require("luatrace")
--   --luatrace.tron()
--   
--   -- (2) update evaluation counter
--   state.evalCounter = state.evalCounter + 1
--   
--   -- (3) learning rate decay (annealing)
--   local clr = lr / ( (1.0 + nevals)^(gamma) )
--   
--   
--   -- (5) save old parameter
--   local xOne = torch.Tensor( p + 1 )
--   local y = torch.Tensor( p )
--   y = dfdx
--   xOne[{{1,p}}] = state.parametersSlow
--   xOne[ p + 1 ] = 1.0
--   
--   
--   -- (6) parameter update
--   if state.evalCounter > state.sgdSteps then
--      Gy = svdMatrix.mv(G, y)
--      Gty = svdMatrix.mv(Gt, y)
--      state.parametersSlow:add( -clr/2.0 , Gy )
--      state.parametersSlow:add( -clr/2.0 , Gty )
--   else
--      state.parametersSlow:add( -clr , y )
--   end
--     
--   x:mul( (sgdolsState.evalCounter -1)/sgdolsState.evalCounter )
--   x:add( sgdolsState.evalCounter , state.parametersSlow )
--   
--   -- (7) rank one update of matrices
--   uno = 1.0
--   local Px = torch.Tensor( p + 1 )
--   Px = svdMatrix.mv(P, xOne)
--   --[[
--   print( 'xOne norm : ', xOne:norm(2) )
--   print( 'y norm : ', y:norm(2) )
--   print( 'Px norm : ', Px:norm(2) )
--   print( 'state.parametersSlow norm : ', state.parametersSlow:norm(2) )
--   --]]
--   
--   
--   
--   b = uno + xOne:dot(Px)
--   alpha = uno/b
--   local u = torch.Tensor(p + 1)
--   u = Px:narrow(1, 1, p)
--   u:mul(alpha)
--   local v = torch.Tensor(p)
--   v = y:clone()
--   B:t()
--   local Btx = torch.Tensor( p )
--   Btx = svdMatrix.mv( B , xOne)
--   --print( 'Btx norm : ', Btx:norm(2) )
--   B:t()
--   
--   Btx:mul(-1.0)
--   v:add( Btx )
--   --print( 'xOne max: ', xOne[1])
--   --print( 'Btx max' , Btx[1])
--   state.B:addr( alpha , Px , v )
--   state.P:addr( -alpha ,Px , Px )
--   
--   local Gu = torch.Tensor( p )
--   Gu = svdMatrix.mv( G , u )
--   local Gv = torch.Tensor( p )
--   G:t()
--   Gv = svdMatrix.mv( G , v )
--   G:t()
--   b = uno + v:dot( Gu )
--   beta =  uno/b
--   
--   state.G:addr(-beta, Gu , Gv )
--   state.Gt:addr(-beta, Gv , Gu )
--   
--   -- stop trace
--   --luatrace.troff()
--   --os.exit()
--   
--   -- return x*, f(x) before optimization
--   return x,{fx}
--end

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 32)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 6)           maximum number of iterations
   --backend                  (default nn)            backend
   --sgdSteps                 (default 19000)     SGD steps before SGDOLS
   
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
      local torchUtil = torch.FloatTensor()
      --print(torchUtil)
      local flip_mask = torchUtil:randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        --print('image : ')
        --print(input[i])
        --if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output = input
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))
      
model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))
   
model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.RReLU())
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))
   
-- Fully Connected Layers   
model:add(nn.SpatialConvolution(256, 1024, 3, 3, 1, 1, 0, 0))
model:add(nn.RReLU())
model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(1024, 1024, 2, 2, 1, 1, 0, 0))
model:add(nn.RReLU())
model:add(nn.Dropout(0.5))
   
model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
model:add(nn.Reshape(10))
model:add(nn.SoftMax())

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()
print('parameters type : \n')
print(type(parameters))


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  local nbPar = parameters:size(1)
  print('<train> nb of parameters : \n')
  print(nbPar)
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.Tensor(opt.batchSize)
  local torchUtil = torch.DoubleTensor()
  print(torchUtil)
  local indices = torchUtil:randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    
    -- Perform SGDOLS step:
    sgdolsState = sgdolsState or {
      numParameters = nbPar,
      learningRate = opt.learningRate,
      rank = opt.rank,
      momentum = opt.momentum,
      sgdSteps = opt.sgdSteps,
      learningRateDecay = 5e-7,
      parametersMean = torch.Tensor(nbPar):zero(),
      evalCounter = 0
      }
    --print('evalCounter : \n')
    --print(sgdolsState.evalCounter)
    if (sgdolsState.evalCounter > (sgdolsState.sgdSteps/opt.batchSize - 1)) then
      print('SGDOLS')
      optim.sgdsvd(feval, parameters, sgdolsState)
    else
      --local neval = sgdolsState.evalCounter
      --local frac = 1/(neval + 1)
      --local parametersAdd = parameters
      --local parametersMean = sgdolsState.parametersMean
      --sgdolsState.parametersMean:add(-frac,parametersMean)
      --sgdolsState.parametersMean:add(frac,parametersAdd)
      --print('SGD')
      optim.sgd(feval, parameters, sgdolsState)
      --if (sgdolsState.evalCounter > (sgdolsState.sgdSteps/opt.batchSize - 1) - 0.5 ) then
      -- optim.sgdolsInit(feval, parameters , sgdolsState)
      --end
    end
    
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 32
  for i=1,provider.testData.data:size(1) - bs,bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    --torch.setdefaulttensortype('torch.FloatTensor')
    --testLogger:plot()
    --torch.setdefaulttensortype('torch.CudaTensor')

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

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end

--luatrace.troff()


