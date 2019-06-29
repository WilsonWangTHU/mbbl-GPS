# generate the config files and run the experiments

date_str=`date`
date_str=${date_str// /-}
date_str=${date_str//:/-}
env_name=$1
# python python/gps/gps_main.py gym_cheetah_mdgps_example 2>&1 | tee output.log
for batch_size in 5000 10000; do
    for rand_seed in 1234 2345 2314 1234 1235; do
        # generate the config files
        exp_name=${date_str}_${env_name}_batch_${batch_size}_seed_${rand_seed}
        cp -r experiments/gym_mdgps_example experiments/${exp_name}_mdgps

        # modify the config files
        sed -i "s/ENV_NAME/${env_name}/g" experiments/${exp_name}_mdgps/hyperparams.py
        sed -i "s/RAND_SEED/${rand_seed}/g" experiments/${exp_name}_mdgps/hyperparams.py
        sed -i "s/TIMESTEPS_PER_BATCH/${batch_size}/g" experiments/${exp_name}_mdgps/hyperparams.py

        # run the experiments
        python python/gps/gps_main.py ${exp_name}_mdgps 2>&1 | tee ./log/${exp_name}_mdgps.log

    done
done
