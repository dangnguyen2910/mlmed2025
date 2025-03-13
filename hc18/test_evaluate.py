import pytest 
import numpy as np 
from .evaluate import pixel_to_mm, approximate_ellipse_circumference
from .hc_dataset import HCDataset

@pytest.mark.skip(reason="Not implemented")
def test_mask_to_hc(): 
    pass 

def test_approximate_ellipse_circumference(): 
    assert np.round(approximate_ellipse_circumference(10, 8), 4) == 28.3617

def test_pixel_to_mm(): 
    dataset = HCDataset("data/hc18/training_set")
    img, gt = dataset[0]
    pixel_size = gt['pixel_size']
    assert pixel_to_mm(300, pixel_size) == 300 * 0.0691358041432

