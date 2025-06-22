import torch
from envs.text2motion_env import Text2MotionEnv
from models.policy_wrapper import PolicyWrapper
from models.text_encoder import load_text_encoder, encode_text


def evaluate(sample_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = load_text_encoder(device)
    text_embedding = encode_text(sample_text, text_encoder).detach().cpu().numpy()

    env = Text2MotionEnv(text_embedding)
    policy = PolicyWrapper(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = policy.act(obs_tensor)
        action_np = action.squeeze().cpu().numpy()
        obs, reward, done, info = env.step(action_np)
        total_reward += reward

    env.render()
    print(f"Total Reward: {total_reward:.2f}")