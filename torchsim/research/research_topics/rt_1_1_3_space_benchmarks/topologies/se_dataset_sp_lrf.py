from torchsim.research.research_topics.rt_1_1_3_space_benchmarks.topologies.se_dataset_ta_lrf import SeDatasetTaLrf


class SeDatasetSpLrf(SeDatasetTaLrf):
    """
    A model which receives data from the SE dataset and learns spatial representation from this.
    """
    def __init__(self,
                 seed: int=None,
                 device: str='cuda',
                 eox: int=2,
                 eoy: int=2,
                 num_cc: int = 100,
                 batch_s=300,
                 tp_learn_period=50,
                 tp_max_enc_seq=1000):

        super().__init__(run_just_sp=True,  # the only change compared to SeDatasetTaLrf is here
                         seed=seed,
                         device=device,
                         eox=eox,
                         eoy=eoy,
                         num_cc=num_cc,
                         batch_s=batch_s,
                         tp_learn_period=tp_learn_period,
                         tp_max_enc_seq=tp_max_enc_seq)