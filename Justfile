

run-mujoco:
    rm -Rf training_videos/*
    rm -Rf data/*
    MUJOCO_GL=egl python train_ppo.py
