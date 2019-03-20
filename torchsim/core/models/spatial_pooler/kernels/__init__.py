from torchsim.core.kernels import load_kernels


sp_process_kernels = load_kernels(__file__, 'sp_processes_kernels', ['sp_processes.cpp',
                                                                     'compute_squared_distances.cu'])
