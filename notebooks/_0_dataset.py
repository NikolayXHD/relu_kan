# %%
from datasets import load_dataset

ds = load_dataset('uoft-cs/cifar10')

# %%
ds

# %%
display(ds['train']['img'][0])
print(ds['train']['label'][0])

print()

# %%
# https://huggingface.co/datasets/uoft-cs/cifar10
label_to_text = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# %%
img = ds['train']['img'][0]

# %%
img

# %%
img_hsv = img.convert('HSV')

# %%
import numpy as np

img_arr = np.array(img_hsv)
print(img_arr.shape)
print(img_arr.dtype)

# %%
import torch

torch.Tensor(np.array(img_hsv.getchannel('H')))

# %%
from matplotlib import pyplot as plt

x = np.arange(256, dtype=np.uint8)
scale = 2 * np.pi / 256

x_cos = np.cos(x * scale)
x_sin = np.sin(x * scale)

fig, ax = plt.subplots(figsize=(14, 3))

ax.plot(x, x_cos, '.-', label='cos', linewidth=1, markersize=2)
ax.plot(x, x_sin, '.-', label='sin', linewidth=1, markersize=2)

ax.grid(which='major', alpha=0.4)
ax.legend()

plt.show()

# %%
type(ds)

# %%
type(ds['train'])

# %%
type(img)

# %%
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-10, 10 + 1)
y = x / (1 + np.exp(-x))

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(x, y, '.-', linewidth=1, markersize=2)
ax.grid(which='major', alpha=0.4)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.show()

# %%
