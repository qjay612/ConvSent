require 'nn'

params = {
    INPUT_CH = 1,
    WORD_VEC_SIZE = 300,
    NUM_PER_KERNEL = 100,
    KERNEL_SIZE = {2,3,4,5,6},
    CLASS_HIDDEN = 200,
    CLASS_NUM = 50
}

conv = nn.ConcatTable()

for k,v in ipairs(params.KERNEL_SIZE) do
    conv:add(
        nn.Sequential()
            :add(
                nn.SpatialConvolution(
                    params.INPUT_CH,
                    params.NUM_PER_KERNEL,
                    params.WORD_VEC_SIZE,
                    v
                )
            )
            :add(nn.Max(2))
            :add(nn.Reshape(NUM_PER_KERNEL))
    )
end

model = nn.Sequential()
    :add(conv)
    :add(nn.JoinTable(1))
    :add(nn.Dropout())
    :add(nn.Linear(#params.KERNEL_SIZE * params.NUM_PER_KERNEL, params.CLASS_HIDDEN))
    :add(nn.Tanh())
    :add(nn.Linear(params.CLASS_HIDDEN, params.CLASS_NUM))
    :add(nn.Tanh())
