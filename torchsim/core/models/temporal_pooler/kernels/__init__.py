from torchsim.core.kernels import load_kernels


tp_process_kernels = load_kernels(__file__, 'tp_processes_kernels', ['tp_processes.cpp',
                                                                     'count_batch_seq_occurrences.cu',
                                                                     'identify_new_seqs.cu',
                                                                     'count_unique_seqs.cu',
                                                                     'compute_seq_likelihoods_clusters_only.cu',
                                                                     'discount_rewards_iterative.cu'])
