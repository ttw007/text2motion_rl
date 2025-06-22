import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gc
import numpy as np
import argparse
from tqdm import tqdm
from utils.text_utils import TextPreprocessor
from utils.motion_utils import normalize_motion, pad_motion

def process_one_line(line):
    parts = line.strip().split("#")
    return parts[0].strip() if len(parts) > 0 else None

def process_sample(text_file, motion_file, tokenizer, max_len):
    motions = np.load(motion_file).astype(np.float64)  # 保留 float64 精度
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    token_list = []
    motion_list = []

    for line in lines:
        text = process_one_line(line)
        if not text:
            continue
        try:
            tokens = tokenizer.tokenize(text, max_length=64)
            motion = normalize_motion(motions)
            motion = pad_motion(motion, max_len)

            token_list.append(tokens["input_ids"].squeeze(0).numpy().astype(np.int32))
            motion_list.append(motion.astype(np.float64))
        except Exception as e:
            print(f"⚠ Error processing line '{line.strip()}': {e}")
            continue

    return token_list, motion_list

def save_batch(output_dir, batch_id, batch_tokens, batch_motions):
    text_path = os.path.join(output_dir, f"texts_part{batch_id}.npy")
    motion_path = os.path.join(output_dir, f"motions_part{batch_id}.npy")
    np.save(text_path, np.array(batch_tokens, dtype=np.int32), allow_pickle=False)
    np.save(motion_path, np.array(batch_motions, dtype=np.float64), allow_pickle=False)
    print(f"✅ Saved batch {batch_id}: {len(batch_tokens)} samples")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = TextPreprocessor()

    text_files = sorted([f for f in os.listdir(args.text_dir) if f.endswith(".txt")])
    gc_interval = 5000
    save_interval = 10000
    sample_counter = 0
    batch_id = 0
    batch_tokens, batch_motions = [], []

    for i, fname in enumerate(tqdm(text_files, desc="Processing Samples")):
        text_path = os.path.join(args.text_dir, fname)
        motion_path = os.path.join(args.motion_dir, fname.replace(".txt", ".npy"))

        if not os.path.exists(motion_path):
            print(f"⚠ Motion file not found for {fname}, skip.")
            continue

        try:
            tokens, motions = process_sample(text_path, motion_path, tokenizer, args.max_len)
            batch_tokens.extend(tokens)
            batch_motions.extend(motions)
            sample_counter += len(tokens)
        except Exception as e:
            print(f"⚠ Error processing {fname}: {e}")
            continue

        if sample_counter >= save_interval:
            save_batch(args.output_dir, batch_id, batch_tokens, batch_motions)
            batch_tokens, batch_motions = [], []
            sample_counter = 0
            batch_id += 1
            gc.collect()

        if i % gc_interval == 0:
            gc.collect()

    if batch_tokens:
        save_batch(args.output_dir, batch_id, batch_tokens, batch_motions)

    print(f"🎉 All done. {batch_id + 1} batches saved to {args.output_dir}")

if __name__ == "__main__":
    class Args:
        text_dir = "D:/study/HumanML3D/HumanML3D/HumanML3D/texts"
        motion_dir = "D:/study/HumanML3D/HumanML3D/HumanML3D/new_joint_vecs"
        output_dir = "./processed_data"
        max_len = 196

    args = Args()
    main(args)
