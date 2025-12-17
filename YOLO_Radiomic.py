import numpy as np
from ultralytics import YOLO
from PIL import Image
import SimpleITK as sitk
from radiomics import featureextractor

def extract_radiomics_features(image_array: np.ndarray, mask_array: np.ndarray, params_file: str = None):

    if image_array.ndim != 2:
        raise ValueError("The image must be a 2D array.")
    if mask_array.ndim != 2:
        raise ValueError("The mask must be a 2D array.")

    image_sitk = sitk.GetImageFromArray(image_array.astype(np.float64))

    # Binarize the mask: all > 0 -> 1
    mask_bin = (mask_array > 0).astype(np.uint8)
    mask_sitk = sitk.GetImageFromArray(mask_bin)

    if params_file:
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    feature_vector = extractor.execute(image_sitk, mask_sitk)

    feature_names = []
    feature_values = []
    
    for key, value in feature_vector.items():
        if not key.startswith(('general_', 'diagnostics_')):
            feature_names.append(key)
            feature_values.append(value)

    return feature_names, feature_values

model = YOLO('runs/segment/train/weights/best.pt')  

results = model("datasets/images/test/98.png")

original_image = Image.open("datasets/images/test/98.png")
original_image_array = np.array(original_image)

if original_image_array.ndim == 3:
    gray_image_array = np.mean(original_image_array, axis=2).astype(np.uint8)
else:
    gray_image_array = original_image_array


# Processing the results
for i, r in enumerate(results):
    # Extracting the radiomic features for each mask
    if r.masks is not None:
        print(f"\nОбнаружено {len(r.masks)} объектов:")
        
        for j, mask in enumerate(r.masks.data):
            # Converting the mask to a numpy array
            mask_array = mask.cpu().numpy()
            
            # If the mask is not binary (for example, it has probability values), we binarize
            if mask_array.max() > 1:
                mask_array = (mask_array > 0.5).astype(np.uint8)
            
            # Resizing the mask to the size of the original image
            if mask_array.shape != gray_image_array.shape:
                # We use PIL to change the size
                mask_resized = Image.fromarray(mask_array)
                mask_resized = mask_resized.resize(
                    (gray_image_array.shape[1], gray_image_array.shape[0]),
                    resample=Image.NEAREST
                )
                mask_array = np.array(mask_resized)
            
            # We check that the mask is not empty
            if np.sum(mask_array) > 0:
                try:
                    feature_names, feature_values = extract_radiomics_features(gray_image_array, mask_array)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print(f"The object mask {j+1} is empty, skipping it")
    else:
        print("Masks are not detected")