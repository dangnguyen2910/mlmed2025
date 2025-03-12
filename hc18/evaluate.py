import numpy as np 
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import mean_absolute_error
from skimage.feature import canny 
import torch 
from torch.utils.data import DataLoader
from .hc_dataset import HCDataset


def main() -> None: 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using ", device)
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 
        'unet',
        in_channels=3, 
        out_channels=1, 
        init_features=32, 
        pretrained=False
    ).to(device).eval()

    model.load_state_dict(torch.load("model/hc_unet.pth", map_location=torch.device(device)))

    test_dataset = HCDataset("data/hc18/test_set")
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    hc_pred_list = []
    hc_true_list = []

    with torch.no_grad(): 
        for i, data in enumerate(test_dataloader): 
            img, gt = data
            mask = gt['mask']
            hc = gt['hc']
            mm_per_pixel = gt['pixel_size']

            img = img.to(device)
            mask = mask.to(device)

            output_mask = model(img).squeeze()
            output_mask = output_mask.cpu().numpy()
            output_mask = (output_mask >= 0.1).astype(int)

            # Save output masks
            fig = plt.figure()
            fig.add_subplot(1,2,1)
            plt.imshow(output_mask, 'gray')
            fig.add_subplot(1,2,2)
            plt.imshow(mask.squeeze().cpu().numpy(), 'gray')
            plt.savefig(f'output/hc18/{i}.jpg')
            plt.close()

            hc_pred = mask_to_hc(output_mask)
            hc_pred = pixel_to_mm(hc_pred, mm_per_pixel)

            hc_pred_list.append(hc_pred)
            hc_true_list.append(hc)

    mae = mean_absolute_error(y_true = hc_true_list, y_pred = hc_pred_list)

    print(f"Mean absolute error: {mae}")


def mask_to_hc(mask) -> float: 
    """ 
    Find the ellipse in the predicted mask then calculate its circumference. 
    
    Parameter: 
    ---
    mask (np.ndarray): Binary mask 
    
    Output: 
    --- 
    Float: Head circumference 
    """
    mask = mask.astype(np.uint8)
    edges = canny(mask, low_threshold=0, high_threshold=1).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours detected: {len(contours)}")
    for contour in contours: 
        if len(contour) < 5: 
            continue

        ellipse = cv2.fitEllipse(contour)
        
        short_axis = ellipse[1][0]
        long_axis = ellipse[1][1]

        break

    head_circumference = approximate_ellipse_circumference(short_axis, long_axis)
    return head_circumference
    

def approximate_ellipse_circumference(a:float, b:float) -> float: 
    """ 
    Approximate circumference of ellipse using Ramanujan formula

    Parameter: 
    --- 
    a (float): length of one axis
    b (float): length of other axis

    Output: 
    ---
    Float: Approximation of ellipse circumference
    """
    l = (a - b)/(a + b)
    return np.pi * (a+b) * (1 + (3 * l**2)/(10 + np.sqrt(4 - 3*l**2)))
    

def pixel_to_mm(value: float, mm_per_pixel: float) -> float: 
    """ 
    Convert value from pixel unit to mm unit 

    Parameter:
    ---
    Value (float): A number in pixel unit
    mm_per_pixel (float): Size of pizel in millimeter

    Output: 
    ---
    Float: value in millimeter unit
    """
    return value * mm_per_pixel


if __name__ == "__main__": 
    main()