import torch
import numpy as np
from _common import BenchmarkDef, op_benchmarks, run_op_benchmark
from _data import InputGenerator, LazyInputDesc, ConcreteInputDesc

# -------------------------------
# torch.repeat (aka ONNX Expand + Tile) benchmark 
# -------------------------------

repeat_configs = [
    BenchmarkDef(
        input_generator = InputGenerator({
            "input_1" : LazyInputDesc(input_shapes=[[1, 64, 16, 32]], dtypes=[np.float32]), 
            "repeats" : ConcreteInputDesc([np.array([2, 1, 16, 1], dtype=np.int32)])
        }),
        variable_arg_names = ['backend', 'mode'],
        variable_arg_vals = [['ortmodule', 'torch'], ['fp16', 'fp32']]
    )
]

@op_benchmarks(repeat_configs)
def bench_repeat(input_1, repeats, backend, mode):
    with torch.no_grad():
        input_data_on_cuda = input_1.cuda()
    class TileNet(torch.nn.Module):
        def forward(self, x):
            x = x.repeat(*repeats)
            return x

    net = TileNet()
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
