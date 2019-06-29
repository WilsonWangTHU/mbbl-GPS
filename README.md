# Guided Policy Search - GPS (MDGPS)
This repo is based on the original [GPS](https://github.com/cbfinn/gps).
We modified the repo to perform benchmarking as part of the [Model Based Reinforcement Learning Benchmarking Library (MBBL)](https://github.com/WilsonWangTHU/mbbl).
Please refer to the [project page](http://www.cs.toronto.edu/~tingwuwang/mbrl.html) for more information.

# Installation
For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).
First install this library as instructed in [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).
Then please go to [MBBL](https://github.com/WilsonWangTHU/mbbl) to install the mbbl package for the environments.

# Run the code

To disable the rendering and run the experiments on headless servers. first run
```
xvfb-run -s "-screen 0 1400x900x24" bash
```

## Perform benchmarking

Please refer to `exp_script/gym_search_2.sh`.

An example to run HalfCheetah looks like this:
```
env_name=gym_cheetah
# python python/gps/gps_main.py gym_cheetah_mdgps_example 2>&1 | tee output.log
for batch_size in 5000; do
    for rand_seed in 1234 2345 2314 1234 1235; do
        # generate the config files
        exp_name=example_${env_name}_batch_${batch_size}_seed_${rand_seed}
        cp -r experiments/gym_mdgps_example experiments/${exp_name}_mdgps

        # modify the config files
        sed -i "s/ENV_NAME/${env_name}/g" experiments/${exp_name}_mdgps/hyperparams.py
        sed -i "s/RAND_SEED/${rand_seed}/g" experiments/${exp_name}_mdgps/hyperparams.py
        sed -i "s/TIMESTEPS_PER_BATCH/${batch_size}/g" experiments/${exp_name}_mdgps/hyperparams.py

        # run the experiments
        python python/gps/gps_main.py ${exp_name}_mdgps 2>&1 | tee ./log/${exp_name}_mdgps.log

    done
done
```

The configuration can be changed by modifying the template config file `experiments/gym_mdgps_example`.
