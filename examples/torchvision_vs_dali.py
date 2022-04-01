# install DALI
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# prepare data
# wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
# tar zxf imagenette2.tgz

import numpy as np
from sklearn.preprocessing import LabelEncoder
from imageio import imread
import PIL
from time import perf_counter
import os, glob
import torch
import torchvision
from torchvision import models, transforms, datasets
import tqdm

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn


valdir = 'imagenette2/val'
resample_size = 256
crop_size = 224
batch_size = 8
num_workers = 1
local_rank = 0
world_size = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.Resize(resample_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize])

device = torch.device('cuda')

model = models.resnet50()
model = model.eval().to(device)

ds = datasets.ImageFolder('imagenette2/val', transform)
dl = torch.utils.data.DataLoader(ds,
                                 batch_size=batch_size,
                                 num_workers=num_workers)

print(len(ds), len(dl))

with torch.no_grad():
  for x, y in tqdm.tqdm(dl):
    _ = model(x.to(device))


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, dali_cpu=False):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=0,
                                     num_shards=1,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    images = fn.decoders.image(images,
                                device=decoder_device,
                                output_type=types.RGB)
    images = fn.resize(images,
                        device=dali_device,
                        size=size,
                        mode="not_smaller",
                        interp_type=types.INTERP_TRIANGULAR)
    mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


pipe = create_dali_pipeline(batch_size=batch_size,
                            num_threads=num_workers,
                            device_id=local_rank,
                            seed=12 + local_rank,
                            data_dir=valdir,
                            crop=crop_size,
                            size=resample_size,
                            dali_cpu=False)
pipe.build()
val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

with torch.no_grad():
  for batch in tqdm.tqdm(val_loader, total=len(val_loader)):
    _ = model(batch[0]['data'])