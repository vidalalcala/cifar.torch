ProFi = require 'ProFi'
ProFi:start()
dofile('train.lua')
ProFi:stop()
ProFi:writeReport( 'MyProfilingReport.txt' )
