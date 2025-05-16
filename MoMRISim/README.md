# MoMRISim Training

## Dataset Generation

We use the [Calgary Campinas Brain MRI Dataset (CC59)](https://portal.conp.ca/dataset?id=projects/calgary-campinas#) \[1] to generate the training data. Run:

```bash
python -m MoMRISim.motion_triplet_generation
```

## Training

We train MoMRISim using [DreamSim](https://github.com/ssundaram21/dreamsim). Steps:

1. **Clone & install DreamSim**

   ```bash
   git clone https://github.com/ssundaram21/dreamsim.git
   cd dreamsim
   pip install -e .
   ```
2. **Replace & rename the dataset loader**

   Copy the motion loader into DreamSim and rename it to `dataset.py`:

    ```bash
    cp ../MoMRISim/dataset/motion_dataset.py dreamsim/datasets/
    ```
3. **Launch training**

    ```bash
    python train.py --config ../MoMRISim/checkpoints/config.yaml
    ```


## References
- [1] Souza et al.  "An Open, Multi-Vendor, Multi-Field-Strength Brain MR Dataset and Analysis of Publicly Available Skull Stripping Methods Agreement". In: *NeuroImage* (2018).
- [2] Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., & Isola, P. (2023). Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344.
