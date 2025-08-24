# Cross-Domain Random Pretraining With Prototype for Reinforcement Learning



![Example Image](pipeline.png)

## Instruction

Enter the repository and use conda to create a environment.
```
cd CRPTpro

conda env create -f conda_env.yml
```

Use tmux to create a terminal (optional) and then enter the created conda environment:
```
tmux

conda activate CRPTpro
```


Run the experiments. 

```
python train.py
```
The data collection, pre-training, and downstream RL are all included.



## Citation


```
@article{CRPTpro,
  title={Cross-domain random pretraining with prototypes for reinforcement learning},
  author={Liu, Xin and Chen, Yaran and Li, Haoran and Li, Boyu and Zhao, Dongbin},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgement
This implementation is built on [Proto-RL](https://github.com/denisyarats/proto).
