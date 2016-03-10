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

if false then
    batch_dataset = nn.pnn.datasetBatch(dataset, 8)

    gpuTable = {1,2,3,4}
    smodel = nn.BatchTable(model)
    tmpCri = nn.CrossEntropyCriterion(torch.Tensor(dataList))
    tmpCri.nll.sizeAverage = false
    criterion = nn.BatchTableCriterion(tmpCri)

    optim_state = {
        learningRate = 0.01,
        learningRateDecay = 1e-5,
        momentum = 0.5
    }

    trainer = nn.MultiGPUTrainer(smodel, criterion, optim.sgd, optim_state, gpuTable)
    trainer:train(batch_dataset, 1000)
else
    cutorch.setDevice(6)
    dataset = pnn.recursiveCuda(dataset)
    function dataset:size() return #dataset end

    model:cuda()
    criterion = nn.CrossEntropyCriterion(torch.Tensor(dataList))
    criterion.sizeAverage = false
    criterion:cuda()

    trainer = nn.StochasticGradient(model, criterion)

    trainer.learningRate = 0.01
    trainer.maxIteration = 25
    trainer:train(dataset)
    torch.saveobj('nobatch_model', model)
end
