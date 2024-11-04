import torch
import torch_fidelity
import lpips
import os
from PIL import Image

ORIGINAL_DATASET_PATH = './datasets/original_21_9'
GENERATED_DATASET_PATH = './datasets/generated_21_9'


def calculate_metrics():
    return torch_fidelity.calculate_metrics(
        input1=ORIGINAL_DATASET_PATH,
        input2=GENERATED_DATASET_PATH,
        cuda=True if torch.cuda.is_available() else False,
        isc=False, fid=True, kid=False, mmd=False, prd=False, verbose=True
    )


def calculate_perceptual_similarity():
    model = lpips.LPIPS(net='alex')  # Choices: 'alex', 'vgg', 'squeeze'
    if torch.cuda.is_available():
        model = model.cuda()

    original_dataset = os.listdir(ORIGINAL_DATASET_PATH)
    generated_dataset = os.listdir(GENERATED_DATASET_PATH)

    lpips_scores = []
    for original_img, generated_img in zip(original_dataset, generated_dataset):
        original_img = Image.open(os.path.join(ORIGINAL_DATASET_PATH, original_img))
        generated_img = Image.open(os.path.join(GENERATED_DATASET_PATH, generated_img))

        if torch.cuda.is_available():
            original_img, generated_img = original_img.cuda(), generated_img.cuda()

        lpips_score = model(original_img, generated_img).item()
        lpips_scores.append(lpips_score)

    # Return average LPIPS score across all image pairs
    return sum(lpips_scores) / len(lpips_scores)


if __name__ == "__main__":
    metrics = calculate_metrics()
    perceptual_similarity = calculate_perceptual_similarity()

    with open('metrics.txt', 'w') as f:
        f.write('EVALUATION METRICS\n\n')

        f.write('Original dataset: %s\n' % ORIGINAL_DATASET_PATH)
        f.write('Generated dataset: %s\n\n' % GENERATED_DATASET_PATH)

        f.write(metrics)
        f.write('\n\n')

        f.write(perceptual_similarity)
        f.write('\n')


