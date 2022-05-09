import torch
x = torch.randn((1, 1), requires_grad=True)
with torch.autograd.profiler.profile(enabled=True) as prof:
    for _ in range(100):  # any normal python code, really!
        y = x ** 2

print(prof.key_averages().table(sort_by="self_cpu_time_total"))

