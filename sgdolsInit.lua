----------------------------------------------------------------------
-- A plain initialization of SGD with OLS estimate of the Hessian
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.learningRate      : learning rate
--   state.P                 : p x p matrix
--   state.B                 : p x (p+1) matrix (OLS beta)
--   state.G                 : p x p matrix (OLS inverse beta)
--   state.X                 : p x n matrix with the initial points, one per column
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function, evaluated before the update
--
-- (Jose V. Alcala-Burgos, 2013)
--
function optim.sgdolsInit(opfunc, x , state)
   -- print
   print('\n','<sgdols> SGDOLS is on !!','\n')
   local p = state.numParameters
   local r = state.rank
   
   print('<sgdols> Hessian matrix initialized as the identity')
   state.P = svdMatrix:new()   
   state.P:ones(p + 1, p + 1)
   state.P:cuda()

   state.B = svdMatrix:new()
   state.B:zero(p + 1, 1, p)
   state.B:cuda()
   
   state.G = svdMatrix:new()   
   state.G:ones(p, p)
   state.G:cuda()
   
   state.Gt = svdMatrix:new()   
   state.Gt:ones(p, p)
   state.Gt:cuda()
   
   state.parametersSlow = torch.Tensor(p)
   state.parametersMean = torch.Tensor(p)
   
   state.parametersSlow[{}] = x
   state.parametersMean[{}] = x

   -- return x*, f(x) before optimization
   fx = opfunc(x)
   
   -- Start profiler
   --luatrace = require("luatrace")
   --luatrace.tron()
   
   return x,{fx}
end
