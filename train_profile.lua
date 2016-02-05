profiler = require 'ProFi'
profiler:start()
dofile('train.lua')
profiler:stop()
profiler:writeReport('report_train.txt')