import torch
from torchgeo.trainers import AutoregressionTask

ckpt_path = ""
num_past_steps = 10
input_size = 1

model = AutoregressionTask.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    teacher_force_prob=False)

example_past = torch.randn(1, num_past_steps, input_size)

model.to_torchscript(method="trace", example_inputs=example_past, file_path="model.pt")