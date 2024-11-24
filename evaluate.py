import torch
import torch_fidelity
import lpips
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim

ORIGINAL_DATASET_PATH = './datasets/original_21_9'
GENERATED_DATASET_PATH = './datasets/generated_21_9'

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def calculate_metrics():
    return torch_fidelity.calculate_metrics(
        input1=ORIGINAL_DATASET_PATH,
        input2=GENERATED_DATASET_PATH,
        cuda=True if torch.cuda.is_available() else False,
        isc=True, fid=True, kid=True, verbose=True
    )

def root_mean_squared_error():
    original_dataset = os.listdir(ORIGINAL_DATASET_PATH)
    generated_dataset = os.listdir(GENERATED_DATASET_PATH)

    rmse_results = []
    for original_img, generated_img in zip(original_dataset, generated_dataset):
        original_img = np.asarray(Image.open(os.path.join(ORIGINAL_DATASET_PATH, original_img)))
        generated_img = np.asarray(Image.open(os.path.join(GENERATED_DATASET_PATH, generated_img)))

        rmse_results.append(rmse(generated_img, original_img))

    return sum(rmse_results) / len(rmse_results)


def structural_similarity_index():
    original_dataset = os.listdir(ORIGINAL_DATASET_PATH)
    generated_dataset = os.listdir(GENERATED_DATASET_PATH)

    ssim_results = []
    for original_img, generated_img in zip(original_dataset, generated_dataset):
        original_img = np.asarray(Image.open(os.path.join(ORIGINAL_DATASET_PATH, original_img)))
        generated_img = np.asarray(Image.open(os.path.join(GENERATED_DATASET_PATH, generated_img)))

        ssim_results.append(ssim(generated_img, original_img, multichannel=True, channel_axis=2))

    return sum(ssim_results) / len(ssim_results)


def calculate_perceptual_similarity():
    model = lpips.LPIPS(net='alex')  # Choices: 'alex', 'vgg', 'squeeze'
    if torch.cuda.is_available():
        model = model.cuda()

    original_dataset = os.listdir(ORIGINAL_DATASET_PATH)
    generated_dataset = os.listdir(GENERATED_DATASET_PATH)

    lpips_scores = []
    for original_img, generated_img in zip(original_dataset, generated_dataset):
        original_img = transforms.ToTensor()(Image.open(os.path.join(ORIGINAL_DATASET_PATH, original_img)))
        generated_img = transforms.ToTensor()(Image.open(os.path.join(GENERATED_DATASET_PATH, generated_img)))

        if torch.cuda.is_available():
            original_img, generated_img = original_img.cuda(), generated_img.cuda()

        lpips_score = model(original_img, generated_img).item()
        lpips_scores.append(lpips_score)

    # Return average LPIPS score across all image pairs
    return sum(lpips_scores) / len(lpips_scores)


if __name__ == "__main__":
    metrics = calculate_metrics()
    perceptual_similarity = calculate_perceptual_similarity()
    rmse_result = root_mean_squared_error()
    ssim_result = structural_similarity_index()

    with open('metrics.txt', 'w') as f:
        f.write('EVALUATION METRICS\n\n')

        f.write('Original dataset: %s\n' % ORIGINAL_DATASET_PATH)
        f.write('Generated dataset: %s\n\n' % GENERATED_DATASET_PATH)

        f.write('Metrics\n')
        f.write(str(metrics))
        f.write('\n\n')

        f.write('Perceptual Similarity\n')
        f.write(str(perceptual_similarity))
        f.write('\n\n')

        f.write('RMSE\n')
        f.write(str(rmse_result))
        f.write('\n\n')

        f.write('SSIM\n')
        f.write(str(ssim_result))
        f.write('\n\n')




