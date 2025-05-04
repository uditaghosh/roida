# [Robust Offline Imitation Learning from Diverse Auxiliary Data](https://arxiv.org/pdf/2410.03626v2)

1. Create the dataset

   ``` bash create_complete_dataset_and_store.sh ```
2. Learn the discriminator

   ``` bash discriminate_expert_subexpert.sh ```
3. ROIDA

   ``` bash algo.sh --env_name $env_name --exp_num $number_of_expert ```
