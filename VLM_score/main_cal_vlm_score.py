'''
Code for registration of all reconstructions
'''
import os
import argparse
import torch
import json
import openai
from vlm_score_utils import image_generation, chatgpt_scoring

def main(recon_save_path: str, image_save_base_path: str, score_save_path: str, open_ai_api: str):
    openai.api_key = open_ai_api
    severity_level_score = {'mo motion':0, 'mild':1, 'moderate':2, 'severe':3}
    image_save_path = os.path.join(image_save_base_path, 'gpt_input')
    baseline_list = ['AltOpt','MotionTTT','stacked_unet']
    score_record = {}
    for baseline in baseline_list:
        score_record[baseline] = {}
        for i in range(8):
            for j in range(3):
                scan_id = f"S{i+1}_{j+1}"
                recon_volume = torch.load(os.path.join(recon_save_path,baseline, f"{scan_id}.pt"), map_location='cpu')
                image_generation(recon_volume, os.path.join(image_save_path, baseline, f"{scan_id}.png"))

                # GPT for scoring
                image_path = os.path.join(image_save_path, baseline, f"{scan_id}.png")
                gpt_output_save_path = os.path.join(image_save_base_path, 'gpt_output', baseline,scan_id)
                # Create the directory if it doesn't exist
                os.makedirs(gpt_output_save_path, exist_ok=True)
                severity_list = []
                for num_exp in range(5):
                    for attempt in range(3):
                        try:
                            sl = chatgpt_scoring(image_path,gpt_output_save_path, 0.5, num_exp)
                            severity_list.append(severity_level_score[sl.lower()])
                            break
                        except Exception as e:
                            print(f"Attempt {attempt + 1} failed: {e}")          
                score_record[baseline][scan_id] = sum(severity_list) / len(severity_list)
    os.makedirs(args.score_save_path, exist_ok=True)
    with open(os.path.join(args.score_save_path, f"vlm_scores.json"), 'w') as f:
            json.dump(score_record, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register all reconstruction volumes")
    parser.add_argument("--recon_save_path", type=str, required=True, help="Path to reconstruction volumes")
    parser.add_argument("--image_save_base_path", type=str, required=True, help="Base folder to save gpt input/output")
    parser.add_argument("--score_save_path", type=str, required=True, help="Path for saving VLM scores")
    parser.add_argument("--openai_api", type=str, required=True, help="Your OpenAI API key")
    args = parser.parse_args()

    main(args.recon_save_path, args.image_save_base_path, args.score_save_path, args.openai_api)