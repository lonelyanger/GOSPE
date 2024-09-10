# GOSPE
source code of GOSPE

## Installation instructions

### Install StarCraft II

Set up StarCraft II and SMAC:

```bash
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over. You may also need to persist the environment variable `SC2PATH` (e.g., append this command to `.bashrc`):

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

### Install Python environment

Install Python environment with conda:

```bash
conda create -n gospe python=3.10 -y
conda activate gospe
pip install -r requirements.txt
```

### Configure SMAC package

We create the additional map following [ODIS](https://github.com/LAMDA-RL/ODIS), for which we're very grateful. Here are a simple script to make some modifications in `smac` and copy additional maps to StarCraft II installation. Please make sure that you have set `SC2PATH` correctly.

```bash
git clone https://github.com/oxwhirl/smac.git
pip install -e smac/
bash install_smac_patch.sh
```

## Run experiments

You can execute the following command to run GOSPE with a toy task config, which will perform training on a small batch of data:

```bash
python src/main.py --mto --config=gospe --env-config=sc2_offline --task-config=toy --seed=1 --note=toytest
```

The `--task-config` flag can be followed with any existing config name in the `src/config/tasks/` directory, and any other config named `xx` can be passed by `--xx=value`. 

As the dataset is large, we only contain the a toy task config of `3m` medium data in the `dataset` folder from the default code base. The data link to the full dataset is [Google Drive URL](https://drive.google.com/file/d/1yyqMBwZkEV6SIXB7F41Lc9tQeCoq_Nza/view?usp=sharing) and you can substitute the original data with the full dataset. After putting the full dataset in `dataset` folder, you can run experiments in pre-defined task sets like 

```bash
python src/main.py --mto --config=gospe --env-config=sc2_offline --task-config=marine-hard-expert --seed=1 --note=SMACtest
```

All results will be stored in the `results` folder. You can see the console output, config, and tensorboard logging in the cooresponding directory.

## License

Code licensed under the Apache License v2.0.

