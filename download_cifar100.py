#!/usr/bin/env python3
import os
import ssl
import urllib.request
import tarfile

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def download_cifar100():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    target_dir = "./data/CIFAR100"
    target_file = os.path.join(target_dir, "cifar-100-python.tar.gz")
    
    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading CIFAR-100 dataset from {url}...")
    urllib.request.urlretrieve(url, target_file)
    print(f"Downloaded to {target_file}")
    
    print("Extracting files...")
    with tarfile.open(target_file, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print(f"Extracted to {target_dir}/cifar-100-python")
    
    print("Download and extraction complete!")

if __name__ == "__main__":
    download_cifar100()