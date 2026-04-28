import argparse
import os
import json
import gzip
import requests
import random
from tqdm import tqdm

def download_and_sample(category, sample_size=10000, output_dir="data", seed=42):
    """
    Downloads Amazon Review 2023 data from UCSD and performs stratified sampling.
    Uses reservoir sampling for each rating to avoid loading the entire dataset into memory.
    """
    print(f"Starting download and sampling for category: {category}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Correct URL from amazon-reviews-2023.github.io
    url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz"
    
    random.seed(seed)
    
    # We want a balanced sample across ratings (1-5) for "stratified" representation
    target_per_rating = sample_size // 5
    reservoirs = {float(i): [] for i in range(1, 6)}
    counts = {float(i): 0 for i in range(1, 6)}
    
    print(f"Downloading from {url}...")
    
    try:
        # We use stream=True to process the file as it downloads
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Use content-length for progress bar if available
        total_size = int(response.headers.get('content-length', 0))
        
        # gzip.open can take a file-like object from requests.raw
        with gzip.open(response.raw, mode='rt', encoding='utf-8') as f:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing")
            
            for line in f:
                # Update progress bar (approximate since we're reading decompressed lines)
                # But we can track the raw stream if we wrap response.raw
                pbar.update(len(line.encode('utf-8')) // 3) # Approximate compression ratio
                
                try:
                    data = json.loads(line)
                    rating = float(data.get('rating', 0))
                    
                    if rating in reservoirs:
                        counts[rating] += 1
                        # Reservoir sampling for this specific rating
                        if len(reservoirs[rating]) < target_per_rating:
                            reservoirs[rating].append(data)
                        else:
                            j = random.randint(0, counts[rating] - 1)
                            if j < target_per_rating:
                                reservoirs[rating][j] = data
                except (json.JSONDecodeError, ValueError):
                    continue
            pbar.close()
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        print("\nPossible issues:")
        print(f"1. Category '{category}' might be misspelled.")
        print("2. The UCSD server might be down or changed the URL.")
        print("Available categories include: All_Beauty, Books, Electronics, Home_and_Kitchen, etc.")
        return

    # Combine all reservoirs
    all_sampled = []
    for r in reservoirs.values():
        all_sampled.extend(r)
    
    # Shuffle the final list to mix ratings
    random.shuffle(all_sampled)
    
    output_file = os.path.join(output_dir, f"amazon_reviews_{category}_sample_{len(all_sampled)}.jsonl")
    
    print(f"Saving {len(all_sampled)} samples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nSample counts per rating:")
    for rating in sorted(counts.keys()):
        print(f"Rating {rating}: Total Processed: {counts[rating]}, Final Sampled: {len(reservoirs[rating])}")
        
    print(f"\nSuccessfully saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and sample Amazon Review 2023 data from UCSD repo.")
    parser.add_argument("--category", type=str, default="All_Beauty", help="Category name (e.g., All_Beauty, Books, Electronics)")
    parser.add_argument("--size", type=int, default=10000, help="Total number of samples to collect (aims for balanced rating distribution)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save the sampled data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()
    
    download_and_sample(
        category=args.category,
        sample_size=args.size,
        output_dir=args.output_dir,
        seed=args.seed
    )