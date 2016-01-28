-- BLAS functions for SVD in torch

svdMatrix = {}
-- a class for the svd decomposition M = U S V^T

function svdMatrix:new(M)
   M = M or {}
   setmetatable(M, self)
   self.__index = self
   M.eps = 0.0000000000000000001
   return M   
end

function svdMatrix:eye(n, r, m)
   self.nbRows = n
   self.rank = r
   self.nbCols = m
   
   torch.setdefaulttensortype('torch.FloatTensor')
   
   self.U = torch.cat( torch.zeros(self.nbRows - self.rank, self.rank), torch.eye(self.rank, self.rank) , 1 )
   self.V = torch.cat( torch.zeros(self.nbCols - self.rank, self.rank), torch.eye(self.rank, self.rank) , 1 )
   self.S = torch.eye(self.rank, self.rank)
   
   torch.setdefaulttensortype('torch.CudaTensor')
   
end

function svdMatrix:ones(n, m)
   -- rank one matrix with eigenvector proportional to (1,1,...,1)^T
   self.nbRows = n
   self.rank = 1
   self.nbCols = m
   
   torch.setdefaulttensortype('torch.FloatTensor')
   
   self.U = torch.ones(self.nbRows, 1)
   self.U:mul(1.0 / self.U:norm(2))
   self.V = torch.ones(self.nbCols, 1)
   self.V:mul(1.0 / self.V:norm(2))
   self.S = torch.eye(self.rank, self.rank)
   
   torch.setdefaulttensortype('torch.CudaTensor')
   
end

function svdMatrix:zero(n, r, m)
   -- zero matrix represented with r zero eigenvalues
   self.nbRows = n
   self.rank = r
   self.nbCols = m
   
   torch.setdefaulttensortype('torch.FloatTensor')
   
   self.U = torch.eye(self.nbRows, self.rank)
   self.V = torch.eye(self.nbCols, self.rank)
   self.S = torch.Tensor(self.rank, self.rank)
   self.S:zero()
   
   torch.setdefaulttensortype('torch.CudaTensor')
end

function svdMatrix:load(U, S ,V)
   -- Loads M = U S V^T 
   self.nbRows = U:size(1)
   self.rank = U:size(2)
   self.nbCols = V:size(1)
   self.U = U
   self.V = V
   self.S = S
end

function svdMatrix:print()
   print('U : ', self.U)
   print('S : ', self.S)
   print('V : ', self.V)
   print('M_full : ', self.M)
end

function svdMatrix:full()
   -- Saves the full matrix in M
   self.M = self.U * self.S * (self.V:t())
end

function svdMatrix:cuda()
   -- Move matrices to GPU, keep M in the CPU
   self.U = self.U:cuda()
   self.S = self.S:cuda()
   self.V = self.V:cuda()
end

function svdMatrix.mv( A, x)
   -- Returns y = A * x 
   local y = torch.Tensor(A.nbRows)
   local z = torch.Tensor(A.rank)
   local w = torch.Tensor(A.rank)
   w:zero()
   w:addmv(A.V:t(), x)
   z:zero()
   z:addmv(A.S, w)
   y:zero()
   y:addmv(A.U, z)
   return y
end

function svdMatrix:addr(alpha, x, y)
   -- Calculates the svd decomposition after the rank one 
   -- update M = M + alpha * x y^T
  
   local p = x:clone()
   local q = y:clone()
   
   local pMat = torch.Tensor(x:size(1), 1)
   local qMat = torch.Tensor(y:size(1), 1)
   
   pMat[{{}, 1}] = p
   qMat[{{}, 1}] = q
   
   --if a:norm(2) < self.eps then
   --   print(' the rank one update is small')
   --   return
   --end
   
   --if b:norm(2) < self.eps then
   --   print(' the rank one update is small')
   --   return
   --end
   
   local m = torch.Tensor(self.rank)
   m:zero()
   m:addmv(self.U:t(), p)
   local n = torch.Tensor(self.rank)
   n:zero()
   n:addmv(self.V:t(), q)
   
   --local p = a:clone()
   p:addmv(-1.0, self.U, m)
   --local q = b:clone()
   q:addmv(-1.0, self.V, n)
   
   --if p:norm(2) < self.eps then
   --   print(' the rank one update is small')
   --   return
   --end
   
   --if q:norm(2) < self.eps then
   --   print(' the rank one update is small')
   --   return
   --end

   local pNorm = torch.Tensor(1, 1)
   local qNorm = torch.Tensor(1, 1)
   pNorm:mm(pMat:t(), pMat)
   qNorm:mm(qMat:t(), qMat)
   local pNormNb = pNorm[1][1]
   local qNormNb = qNorm[1][1]
   pNormNb = math.sqrt(pNormNb)
   qNormNb = math.sqrt(qNormNb)
   pNorm[1][1] = pNormNb  
   qNorm[1][1] = qNormNb
   
   mOne = torch.Tensor(self.rank + 1)
   mOne[{{1, self.rank}}] = m
   mOne[self.rank + 1] = pNorm
   mOne:mul(alpha)
   
   nOne = torch.Tensor(self.rank + 1)
   nOne[{{1, self.rank}}] = n
   nOne[self.rank + 1] = qNorm
   
   p:mul(1.0 / pNormNb)
   q:mul(1.0 / qNormNb)
   
   local K = torch.Tensor(self.rank + 1, self.rank + 1)
   
   K:zero()
   K[{{1,self.rank},{1,self.rank}}] = self.S
   K:addr(mOne, nOne)
   
   K_cpu = K:float()
   
   torch.setdefaulttensortype('torch.FloatTensor')
   u, s, v = torch.svd(K_cpu)
   torch.setdefaulttensortype('torch.CudaTensor')
   
   
   -- Update with rank increase, rank <- rank + 1 .
   --[[
   self.S = torch.diag(s)
   self.U = torch.cat(self.U, p) * u
   self.V = torch.cat(self.V, q) * v
   self.rank = self.rank + 1
   --]]
   
   self.S = torch.diag(s:narrow(1, 1, self.rank)):cuda()
   self.U = (torch.cat(self.U:float(), p:float()) * u:narrow(2, 1, self.rank):float() ):cuda()
   self.V = (torch.cat(self.V:float(), q:float()) * v:narrow(2, 1, self.rank):float() ):cuda()
   
end

function svdMatrix:t()
   -- M <- M^T
   self.U, self.V = self.V, self.U
   self.nbRows , self.nbCols = self.nbCols , self.nbRows
end

-- Test the functions
--[[
n = 3
r = 2
m = 4

A = svdMatrix:new()
A:eye(n, r, m)
A:cuda()
A:full()
print(A.M)
print(A.U)
a = torch.rand(n)
b = torch.rand(m)
a:cuda()
b:cuda()
print(a)
print(b)
A_new = A.M:addr(2.0 , a, b)

A:addr(2.0, a, b)
A:full()
print('A_new : ', A_new)
A:print()
print(a)
print(b)
--]]