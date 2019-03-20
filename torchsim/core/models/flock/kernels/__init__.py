from torchsim.core.kernels import load_kernels

buffer_kernels = load_kernels(__file__, 'buffer_store', ['buffer_store.cpp',
                                                        'buffer_store.cu'])
