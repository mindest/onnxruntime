import torch
import numpy as np
from _common import BenchmarkConfig, op_benchmarks, run_op_benchmark, VisualConfig
from _data import InputGenerator, LazyInputDesc, ConcreteInputDesc

# -------------------------------
# softmax  benchmark 
# -------------------------------

softmax_configs = [
    BenchmarkConfig(
        input_generator = InputGenerator({
            "input_1" : LazyInputDesc(
                input_shapes=[
                    [2400, 128], [2400, 256], [2400, 512], [2400, 1024],
                    [2400, 2048], [2400, 3072], [2400, 4096],
                    [2400, 6144], [2400, 12800]
                ],
                dtypes=[
                    np.float32, np.float32, np.float32, np.float32,
                    np.float32, np.float32, np.float32,
                    np.float32, np.float32,
                ]
            ), 
        }),
        variable_names = ['backend', 'mode'],
        variable_values_pool = [['ortmodule', 'torch'], ['fp16', 'fp32']]
    )
]

visual_config = VisualConfig(pivot_variable_name='backend', pivot_varible_control_value='torch')

@op_benchmarks(softmax_configs, visual_config)
def bench_softmax(input_1, backend, mode):
    with torch.no_grad():
        input_data_on_cuda = torch.from_numpy(input_1).cuda()

    class SoftmaxNet(torch.nn.Module):
        def __init__(self):
            super(SoftmaxNet, self).__init__()
            self.m = torch.nn.Softmax(dim=1)

        def forward(self, x):
            x = self.m(x)
            return x

    net = SoftmaxNet()
    net.cuda()

    if mode == 'fp16':
        net = net.half()
        input_data_on_cuda = input_data_on_cuda.half()

    net.train()

    if backend == 'ortmodule':
        from onnxruntime.training.ortmodule import ORTModule
        net = ORTModule(net)
        # gbps = lambda ms: (2 * a.numel() * a.element_size() * 1e-9) / (ms * 1e-3)
        return run_op_benchmark(lambda: net(input_data_on_cuda))
    elif backend == 'torch':
        return run_op_benchmark(lambda: net(input_data_on_cuda))
