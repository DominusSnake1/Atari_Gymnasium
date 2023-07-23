import gymnasium as gym
import Agents.Agents as agent
from stable_baselines3 import A2C


# def start_CartPole():
#     env = gym.make("CartPole-v1", render_mode="rgb_array")
#
#     model = A2C("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10_000)
#
#     vec_env = model.get_env()
#     obs = vec_env.reset()
#     for i in range(1000):
#         action, _state = model.predict(obs, deterministic=True)
#         obs, reward, done, info = vec_env.step(action)
#         vec_env.render("human")


def testGameWithRandomAgent(game, episodes):
    env = gym.make(game, render_mode="human")

    for ep in range(1, episodes+1):
        env.reset()
        done = False
        score = 0

        while not done:
            action = agent.randomAgent(env)
            _, reward, done, _, _ = env.step(action)
            score += reward
            env.render()

        print(f"Episode {ep}, Score: {score}")

    env.close()