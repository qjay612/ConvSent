dofile './ConvSent.lua'

require 'pnn'
require 'optim'

dataset = torch.load('./train_dataset')

dataList = {}
for i = 1,#dataset do
    local size = dataset[i][1][1]:size()
    if size[1] < params.KERNEL_SIZE[#params.KERNEL_SIZE] then
        local tmp = torch.zeros(params.KERNEL_SIZE[#params.KERNEL_SIZE], size[2])
        tmp:narrow(1,1,size[1]):copy(dataset[i][1][1])
        dataset[i][1][1] = tmp
    end
    size = dataset[i][1][1]:size()
    dataset[i][1] = dataset[i][1][1]:reshape(1,size[1],size[2])
    if dataList[dataset[i][2]] then
        dataList[dataset[i][2]] = dataList[dataset[i][2]] + 1
    else
        dataList[dataset[i][2]] = 1
    end
end

for i = 1,#dataList do
    dataList[i] = 50 / dataList[i]
end

model:training()
criterion = nn.CrossEntropyCriterion(torch.Tensor(dataList))
criterion.sizeAverage = false

local state = {
    learningRate = 0.01,
    learningRateDecay = 0.0002,
    momentum = 0.9,
    wd = 16
}
local trainer = nn.MultiGPUTrainer(model, criterion, optim.sgd, state, {1,2})
trainer:train(dataset, 50, 50)
