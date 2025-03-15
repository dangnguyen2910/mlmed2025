import pytest
from .hc_dataset import HCDataset
import torch
import torchvision.transforms.v2 as v2

@pytest.fixture(scope='session')
def dataset(): 
    dataset = HCDataset("data/hc18/training_set")
    return dataset

def test_getitem_image_size(dataset): 
    image, gt = dataset[0]
    assert image.size() == torch.Size([3, 640, 640])

def test_getitem_mask_size(dataset): 
    image, gt = dataset[0]
    assert gt['mask'].size() == torch.Size([1, 640, 640])

def test_getitem_binary_mask(dataset): 
    image, gt = dataset[0]
    assert torch.equal(torch.unique(gt['mask']), torch.tensor([0,1]))

def test_get_hc(dataset): 
    _, gt = dataset[0]
    hc = dataset.get_head_circumference("000_HC.png")
    assert hc == 44.3
    assert gt['hc'] == 44.3

def test_get_pixel_size(dataset): 
    _, gt = dataset[0]
    pixel_size = dataset.get_pixel_size("000_HC.png")
    assert pixel_size == 0.0691358041432
    assert gt['pixel_size'] == 0.0691358041432