import cv2
import numpy as np
from PIL import Image
import io
import time
from torchvision.io import decode_jpeg
import torch

def generate_sample_frame(width=1920, height=1080):
    """Generate a sample frame similar to what you'd get from kafka"""
    # Create a sample image
    img = np.ones((height, width, 3), dtype=np.uint8) * 40
    # Encode it to jpg bytes like it would be in kafka
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def method1_pil_numpy():
    """Current method using PIL + numpy"""
    frame = generate_sample_frame()
    return np.array(Image.open(io.BytesIO(np.frombuffer(frame, np.uint8))))

def method2_direct_cv2():
    """Direct CV2 decoding with RGB conversion"""
    frame = generate_sample_frame()
    bgr = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def method3_torchvision(batch_size=16):
    batch = []
    for _ in range(batch_size):
        frame = generate_sample_frame()
        batch.append(torch.frombuffer(bytearray(frame), dtype=torch.uint8))
    images = torch.stack(decode_jpeg(batch, device='cuda')).permute(0, 2, 3, 1)
    return images.cpu().numpy()

def analyze_differences(img1, img2):
    """Detailed analysis of differences between images"""
    if img1.shape != img2.shape:
        print(f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}")
        return
    
    print(f"\nImage Analysis:")
    print(f"Shape: {img1.shape}")
    print(f"img1 dtype: {img1.dtype}, img2 dtype: {img2.dtype}")
    print(f"img1 value range: [{img1.min()}, {img1.max()}]")
    print(f"img2 value range: [{img2.min()}, {img2.max()}]")
    
    # Convert both to RGB for comparison
    if len(img1.shape) == 3:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.shape[2] == 3 else img1
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.shape[2] == 3 else img2
    else:
        img1_rgb = img1
        img2_rgb = img2
    
    # Calculate differences
    diff = cv2.absdiff(img1_rgb, img2_rgb)
    num_diff_pixels = np.count_nonzero(diff)
    total_pixels = img1.shape[0] * img1.shape[1] * img1.shape[2]
    
    print(f"\nDifference Analysis:")
    print(f"Number of different pixels: {num_diff_pixels}")
    print(f"Percentage different: {(num_diff_pixels/total_pixels)*100:.2f}%")
    print(f"Max difference value: {diff.max()}")
    

def benchmark_and_compare():
    # Generate sample frame
    print("Generating sample frame...")
    frame = generate_sample_frame()
    
    # Run methods
    print("\nProcessing images...")
    img1 = method1_pil_numpy()
    img2 = method2_direct_cv2()
    img3 = method3_torchvision()
    img3 = img3[0]
    
    # Benchmark timing
    num_iterations = 1000
    print(f"\nBenchmarking {num_iterations} iterations...")
    
    # Time method 1
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = method1_pil_numpy()
    time1 = (time.perf_counter() - start) / num_iterations
    
    # Time method 2
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = method2_direct_cv2()
    time2 = (time.perf_counter() - start) / num_iterations
    
    # Time method 3 (CUDA)
    start = time.perf_counter()
    for _ in range(0, num_iterations, 16):
        _ = method3_torchvision()
    time3 = (time.perf_counter() - start) / num_iterations
    
    # Print timing results
    print("\nTiming Results (seconds per image):")
    print(f"PIL + Numpy method:  {time1:.6f}")
    print(f"Direct CV2 method:   {time2:.6f}")
    print(f"CUDA method:         {time3:.6f}")
    print(f"Speedup factor (CUDA vs PIL): {time1/time3:.2f}x")
    print(f"Speedup factor (CUDA vs CV2): {time2/time3:.2f}x")
    
    # Analyze differences
    analyze_differences(img1, img2)
    analyze_differences(img1, img3)

if __name__ == "__main__":
    benchmark_and_compare()