import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread
from skimage.metrics import structural_similarity
def calculate_psnr(folder1, folder2):
    total_psnr = 0
    total_ssim = 0
    count = 0
    for root1, dirs1, files1 in os.walk(folder1):
        for file1 in files1:
            if file1.lower().endswith('.png'):
                # 构建文件的完整路径
                img1_path = os.path.join(root1, file1)
                # print(img1_path)
                # 找到另一个文件夹中对应的图像
                relative_path = os.path.relpath(img1_path, folder1)
                img2_path = os.path.join(folder2, relative_path)
                # print(img2_path)

                # print(os.path.exists(img2_path))
                # print(img2_path.lower().endswith('.png'))
                if os.path.exists(img2_path) and img2_path.lower().endswith('.png'):
                    # 读取图像
                    img1 = imread(img1_path)
                    img2 = imread(img2_path)
                    # 计算 PSNR
                    psnr = peak_signal_noise_ratio(img1, img2,data_range = 65535)
                    ssim = structural_similarity(img1, img2,data_range = 65535)
                    total_psnr += psnr
                    total_ssim += ssim
                    count += 1
                    print(f"For images {img1_path} and {img2_path}, PSNR: {psnr}")
                    print(f"For images {img1_path} and {img2_path}, ssim: {ssim}")
    if count > 0:
        average_psnr = total_psnr / count
        average_ssim = total_ssim / count
        print(f"Average PSNR: {average_psnr}")
        print(f"Average ssim: {average_ssim}")
        return average_psnr
    else:
        print("No matching PNG images found in both folders.")
        return None
if __name__ == "__main__":
    folder1 = 'data/5000voluneer'  # 替换为第一个文件夹的路径
    folder2 = 'data/5000voluneer-pre'  # 替换为第二个文件夹的路径
    calculate_psnr(folder1, folder2)
