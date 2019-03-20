#include <torch/extension.h>

// forward declaration for func in .cu file
void count_batch_seq_occurrences(at::Tensor cluster_batch,
                                      at::Tensor context_batch,
                                      at::Tensor rewards_punishment_batch,
                                      at::Tensor all_encountered_seqs,
                                      at::Tensor all_encountered_seq_occurrences,
                                      at::Tensor encountered_batch_seq_occurrences,
                                      at::Tensor encountered_batch_context_occurrences,
                                      at::Tensor encountered_batch_rewards_punishments,
                                      at::Tensor newly_encountered_seqs_indicator,
                                      at::Tensor explorations,
                                      at::Tensor actions,
                                      at::Tensor encountered_exploration_attempts,
                                      at::Tensor encountered_exploration_successes,
                                      int flock_size,
                                      int n_cluster_centers,
                                      int seq_length,
                                      int seq_lookbehind,
                                      int used_encountered_seqs,
                                      int max_seqs_in_subbatch,
                                      int context_size,
                                      int n_subbatches,
                                      int max_seqs_in_batch,
                                      int n_providers);


void identify_new_seqs(at::Tensor batch,
                            at::Tensor newly_encountered_seqs_indicator,
                            at::Tensor most_probable_batch_seqs,
                            at::Tensor most_probable_batch_seq_probs,
                            int flock_size,
                            int seq_length,
                            int max_seqs_in_batch,
                            int n_cluster_centers);


void count_unique_seqs(at::Tensor most_probable_batch_seqs,
                            at::Tensor most_probable_batch_seq_probs,
                            at::Tensor newly_encountered_seqs_counts,
                            int flock_size,
                            int seq_length,
                            int max_seqs_in_batch);


void compute_seq_likelihoods_clusters_only(at::Tensor cluster_history,
                                           at::Tensor frequent_seqs,
                                           at::Tensor frequent_occurrences,
                                           at::Tensor seq_likelihoods,
                                           int flock_size,
                                           int n_frequent_seqs,
                                           int seq_lookbehind);


void discount_rewards_iterative(at::Tensor frequent_sequences,
                                at::Tensor seq_probs_priors_clusters_context,
                                at::Tensor current_rewards,
                                at::Tensor influence_model,
                                at::Tensor cluster_rewards,
                                int flock_size,
                                int n_frequent_seqs,
                                int n_cluster_centers,
                                int transition,
                                int n_providers);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "Temporal pooler process kernels"; // optional module docstring
  m.def("count_batch_seq_occurrences", &count_batch_seq_occurrences, "Count batch seq occurrences");
  m.def("identify_new_seqs", &identify_new_seqs, "Identify new seqs kernel");
  m.def("count_unique_seqs", &count_unique_seqs, "Count unique seqs kernel");
  m.def("compute_seq_likelihoods_clusters_only", &compute_seq_likelihoods_clusters_only,
  "Compute seq likelihoods based just on clusters kernel");
  m.def("discount_rewards_iterative", &discount_rewards_iterative, "Computes rewards for following sequences");
}

