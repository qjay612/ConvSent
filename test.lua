dofile('ConvSent.lua')

require 'cutorch'
require 'pnn'

cutorch.setDevice(1)

dataset = torch.load('./test_dataset')

for i = 1,#dataset do
    local size = dataset[i][1][1]:size()
    if size[1] < params.KERNEL_SIZE[#params.KERNEL_SIZE] then
        local tmp = torch.zeros(params.KERNEL_SIZE[#params.KERNEL_SIZE], size[2])
        tmp:narrow(1,1,size[1]):copy(dataset[i][1][1])
        dataset[i][1][1] = tmp
    end
    size = dataset[i][1][1]:size()
    dataset[i][1] = dataset[i][1][1]:reshape(1,size[1],size[2])
end
dataset = pnn.recursiveCuda(dataset)

model = torch.loadobj('model')
model:evaluate()
model:cuda()

correctNum = 0

for i = 1,#dataset do
    output = model:forward(dataset[i][1])
    maxIndex = 0
    maxValue = -1
    for j = 1,output:size()[1] do
        if output[j] > maxValue then
            maxValue = output[j]
            maxIndex = j
        end
    end
    print(maxIndex, dataset[i][2], output[dataset[i][2]])
    if maxIndex == dataset[i][2] then
        correctNum = correctNum + 1
    end
end

print(correctNum / #dataset * 100)
