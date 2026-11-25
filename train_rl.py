"""
Reinforcement Learning Training & Evaluation Script for Waveshare JetRacer.

Trains a PPO agent (via Stable-Baselines3) to drive autonomously around the
test track using either:
- Vector state observation (IMU + velocities + RPLIDAR A1 rangefinder)
- Camera image observation (front CSI camera)

Usage:
    # Train PPO for 50,000 steps with vector observation
    python3 train_rl.py --train --timesteps=50000

    # Evaluate the trained policy
    python3 train_rl.py --eval --model_path=jetracer_ppo.zip

    # Run random baseline policy test
    python3 train_rl.py --random --steps=200
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

# Ensure local module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetracer_env import JetRacerEnv


def train(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("[ERROR] stable-baselines3 is required for training: pip install stable-baselines3")
        sys.exit(1)

    print(f"\n[*] Initializing JetRacerEnv (obs_mode='{args.obs_mode}') for PPO training...")
    env = JetRacerEnv(
        obs_mode=args.obs_mode,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
    )

    policy_type = "MlpPolicy" if args.obs_mode == "vector" else "CnnPolicy"
    print(f"[*] Policy type: {policy_type}")
    print(f"[*] Total timesteps: {args.timesteps:,}")

    model = PPO(
        policy_type,
        env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        device="cpu" if args.obs_mode == "vector" else "auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 5, 2000),
        save_path="./checkpoints/",
        name_prefix="jetracer_ppo",
    )

    print("\n[+] Starting training loop...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    save_path = "jetracer_ppo.zip"
    model.save(save_path)
    print(f"\n[+] Training complete in {elapsed:.1f}s ({args.timesteps / elapsed:.1f} steps/s)!")
    print(f"[+] Model checkpoint saved to: {save_path}")
    env.close()


def evaluate(args):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 is required for evaluation: pip install stable-baselines3")
        sys.exit(1)

    model_path = args.model_path or "jetracer_ppo.zip"
    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint '{model_path}' not found. Train first with --train.")
        sys.exit(1)

    print(f"\n[*] Loading trained PPO agent from: {model_path}")
    model = PPO.load(model_path)

    env = JetRacerEnv(obs_mode=args.obs_mode, frame_skip=args.frame_skip)
    obs, info = env.reset(seed=42)

    print(f"[*] Running {args.eval_episodes} evaluation episode(s)...")
    for ep in range(args.eval_episodes):
        ep_reward = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if terminated or truncated:
                reason = "COLLISION" if info["is_collision"] else ("ROLLOVER" if info["is_rollover"] else "TIMEOUT")
                print(f"  - Episode {ep + 1}: Return = {ep_reward:+6.2f} | Steps = {steps:4d} | Speed = {info['speed_m_s']:.2f} m/s | End: {reason}")
                obs, info = env.reset()
                break

    env.close()


def random_baseline(args):
    print("\n[*] Running Random Baseline Policy...")
    env = JetRacerEnv(obs_mode="vector", frame_skip=args.frame_skip)
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    episodes = 0
    ep_reward = 0.0

    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        total_reward += reward

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1:4d}: speed={info['speed_m_s']:.2f} m/s, cross_track={info['cross_track_error']:+.3f}m, return={ep_reward:+.1f}")

        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()
            ep_reward = 0.0

    print(f"\n[+] Random Baseline finished {args.steps} steps across {episodes} episode(s). Average return: {total_reward / max(1, episodes):.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="JetRacer MuJoCo Reinforcement Learning Pipeline")
    parser.add_argument("--train", action="store_true", help="Train a PPO policy")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained policy checkpoint")
    parser.add_argument("--random", action="store_true", help="Run a random baseline agent")
    parser.add_argument("--obs_mode", choices=["vector", "rgb"], default="vector", help="Observation space mode")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--n_steps", type=int, default=512, help="PPO rollout buffer size per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--frame_skip", type=int, default=10, help="Simulation sub-steps per action (10 = 50Hz control)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max episode length before truncation")
    parser.add_argument("--model_path", type=str, default="jetracer_ppo.zip", help="Path to checkpoint for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--steps", type=int, default=300, help="Number of random steps to run")

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    elif args.random:
        random_baseline(args)
    else:
        # Default behavior: run random baseline test
        print("No mode selected (--train, --eval, or --random). Running random baseline by default.")
        random_baseline(args)


if __name__ == "__main__":
    main()
