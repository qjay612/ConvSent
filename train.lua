dofile './ConvSent.lua'

require 'pnn'

dataset = torch.load('./train_dataset')

dataList = {}
for i = 1,#dataset do
    dataset[i][1] = dataset[i][1][1]
    if dataList[dataset[i][2]] then
        dataList[dataset[i][2]] = dataList[dataset[i][2]] + 1
    else
        dataList[dataset[i][2]] = 1
    end
end

for i = 1,#dataList do
    dataList[i] = 50 / dataList[i]
end

batch_dataset = nn.pnn.datasetBatch(dataset, 8)

gpuTable = {5,6,7,8}
smodel = nn.BatchTable(model)
tmpCri = nn.CrossEntropyCriterion(torch.Tensor(dataList))
tmpCri.nll.sizeAverage = false
criterion = nn.BatchTableCriterion(tmpCri)

adam_state = {}

trainer = nn.MultiGPUTrainer(smodel, criterion, optim.adam, adam_state, gpuTable)
trainer:train(batch_dataset, 1000)
