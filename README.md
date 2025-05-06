# [Robust Offline Imitation Learning from Diverse Auxiliary Data](https://arxiv.org/pdf/2410.03626v2)

1. Create the dataset

   ``` bash create_complete_dataset_and_store.sh ```
2. Learn the discriminator

   ``` bash discriminate_expert_subexpert.sh ```
3. ROIDA

   ``` bash algo.sh ```

You may need to change the hyperparameters like env_name and number of experts as necessary in the bash files.

If you find this helpful, please cite
```
@misc{ghosh2024robustofflineimitationlearning,
      title={Robust Offline Imitation Learning from Diverse Auxiliary Data}, 
      author={Udita Ghosh and Dripta S. Raychaudhuri and Jiachen Li and Konstantinos Karydis and Amit K. Roy-Chowdhury},
      year={2024},
      eprint={2410.03626},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.03626}, 
}
```
