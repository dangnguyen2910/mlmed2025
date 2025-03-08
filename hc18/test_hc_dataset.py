import pytest
from .hc_dataset import HCDataset
import torch
import torchvision.transforms.v2 as v2

@pytest.fixture(scope='session')
def dataset(): 
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640,640)), 
    ])
    dataset = HCDataset("data/hc18/training_set", transform, transform)
    return dataset

def test_getitem_train(dataset): 
    image, gt = dataset[0]
    assert image.size() == torch.Size([3, 640, 640])
    assert gt['mask'].size() == torch.Size([1, 640, 640])


def test_get_hc(dataset): 
    image, gt = dataset[0]
    hc = dataset.get_head_circumference("000_HC.png")
    assert hc == 44.3
    assert gt['hc'] == 44.3