import torch

from _common import BenchmarkDef, op_benchmarks, run_op_benchmark

# -------------------------------
# Softmax benchmark 
# -------------------------------

softmax_configs = [
    BenchmarkDef(
              input_names = ["input_1"],
              input_shapes = [
                  [[2400, 128]], 
                  [[2400, 256]],
                  [[2400, 512]], 
                  [[2400, 1024]], 
                  [[2400, 2048]], 
                  [[2400, 3072]], 
                  [[2400, 4096]], 
                  [[2400, 6144]], 
                  [[2400, 12800]]
                ],
              variable_arg_names = ['backend', 'dtype'],
              variable_arg_vals = [['ortmodule'], ['fp16', 'fp32']],
    )
]

@op_benchmarks(softmax_configs)
def bench_softmax(input_1, backend, dtype, warmup=10, rep=50):
    with torch.no_grad():
        M = input_1[0]
        N = input_1[1]
        input_data_on_cuda = torch.rand(M, N).cuda()

    class SoftmaxNet(torch.nn.Module):
        def __init__(self):
            super(SoftmaxNet, self).__init__()
            self.m = torch.nn.Softmax(dim=1)

        def forward(self, x):
            x = self.m(x)
            return x

    net = SoftmaxNet()
    net.cuda()

    if dtype == 'fp16':
        net = net.half()
        input_data_on_cuda = input_data_on_cuda.half()

    net.train()

    if backend == 'ortmodule':
        from onnxruntime.training.ortmodule import ORTModule
        net = ORTModule(net)
        # gbps = lambda ms: (2 * a.numel() * a.element_size() * 1e-9) / (ms * 1e-3)
        return run_op_benchmark(lambda: net(input_data_on_cuda), warmup_step=warmup, repeat_step=rep)
    elif backend == 'torch':
        return run_op_benchmark(lambda: net(input_data_on_cuda), warmup_step=warmup, repeat_step=rep)
