import logging
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.logger = logging.getLogger("TrainLogger")
        self.logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def log(self, info: dict, step: int):
        for k, v in info.items():
            self.writer.add_scalar(k, v, step)
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in info.items()])
        self.logger.info(f"Step {step} | {log_str}")

    def log_metrics(self, epoch, losses, ep_reward):
        print(f"Epoch {epoch:04d} | Reward: {ep_reward:.2f} | "
          f"Policy Loss: {losses['policy_loss']:.4f} | "
          f"Value Loss: {losses['value_loss']:.4f}")
