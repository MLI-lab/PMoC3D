import numpy as np
import os
import matplotlib.pyplot as plt
import openai
import base64
import json
import re

def normalize_percentile(img, lower_percentile=1, upper_percentile=99.9, clip=True):
    """ Normalization to the lower and upper percentiles 
        Utility functions from:
        https://github.com/melanieganz/ImageQualityMetricsMRI/blob/main/utils/data_utils.py

    """
    img = img.astype(np.float32)
    lower = np.percentile(img, lower_percentile)
    upper = np.percentile(img, upper_percentile)
    img = (img - lower) / (upper - lower)
    if clip:
        img = np.clip(img, 0, 1)
    return img

def image_generation(recon, save_path):
    recon = normalize_percentile(recon.numpy())
    slice_indices = [80, 108, 151]
    views = ['Axial', 'Sagittal', 'Coronal']
    n_slices = len(slice_indices)
    figsize = (n_slices * 4, len(views) * 4)

    fig, axes = plt.subplots(len(views), n_slices, figsize=figsize)
    plt.subplots_adjust(wspace=0.03, hspace=0.04)

    for view_idx, view in enumerate(views):
        for slice_idx_idx, slice_idx in enumerate(slice_indices):
            ax = axes[view_idx, slice_idx_idx]
            if view == 'Axial':
                recon_slice = recon[slice_idx, :, :]
            elif view == 'Sagittal':
                recon_slice = recon[:, :, slice_idx]
            elif view == 'Coronal':
                recon_slice = recon[:, slice_idx, :]
            ax.imshow(recon_slice, cmap='gray')
            ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def chatgpt_scoring(image_path,json_folder, temperature, expert_num):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64_str = base64.b64encode(image_bytes).decode("utf-8")
    motion_artifact_prompt = '''
**Task:**  
Evaluate the severity of motion artifacts in the provided MRI image using a structured and systematic analysis.
---
### **Evaluation Criteria for MRI Image**
- **No Motion Artifact:** No visible motion artifacts; excellent diagnostic quality, and minor reconstruction noise is acceptable.
- **Mild:** The majority of brain details are clearly visible, with only minor artifacts that do not obscure diagnostic structures; minimal diagnostic impact, and minor reconstruction noise is acceptable.
- **Moderate:** Noticeable artifacts that partially obscure critical diagnostic regions; artifacts significantly impact diagnostic interpretation.  
- **Severe:** Brain structures are predominantly obscured by artifacts, with only the general shape discernible; diagnosis is extremely challenging or impossible.
### **Output Template**
**Analyze Brain Structure Visibility**  
- Does the image look very smooth, potentially losing significant detail? *(Important for scoring!)*  
- Are all major brain details visible (gyri, sulci, ventricles)?  
- Do motion artifacts blur or distort critical brain details?  
- Are there regions where brain details are completely lost?
**Assess Artifact Types and Locations**  
- Check for ringing effects (where, how severe).  
- Identify other motion artifacts (streaking, ghosting) and note their severity.
**Oversmooth Assessment**  
    - Does the image look very smooth (like a very high-quality image)?  
    - Are there areas with smooth distortions?  
    - If yes, do you think the image has an oversmoothing problem?
- The primary MRI image shows **[overall assessment]** motion artifacts. The final precise motion artifact level is: [No Motion/Mild/Moderate/Severe]
If the severity level is No Motion/Mild: Re-examine the image. Are all details truly clear? If any structures appear compromised, consider increasing the severity level.
---
### **Conclusion**
- After rethinking, the primary MRI image shows **[overall assessment]** motion artifacts, and the details are **. Given these factors, the final precise motion artifact level is:
Severity Level: [No Motion/Mild/Moderate/Severe]
'''
    response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
        {
                        "role": "system",
                        "content": (
                            "You are an MRI image analysis expert whose task is to evaluate "
                            "the severity of motion artifacts in MRI images. This is not medical advice."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is the MRI image for evaluation :"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64_str}"
                                },
                            },
                            {"type": "text", "text": motion_artifact_prompt},
                        ],
                    }
                ],
                max_tokens=600,  
                temperature=temperature,
            )

    gpt_out = response.choices[0].message.content
    print(gpt_out)
    analysis_text = gpt_out.strip()
    score_line = analysis_text.split("\n")[-1]

    pattern = re.compile(
        r"(?:Severity\s*Level:\s*)?(No\s*Motion|Mild|Moderate|Severe)",
        re.IGNORECASE
    )
    matches = pattern.findall(score_line)
    print(matches)
    output_data = {
        "text": analysis_text,
        "Severity Level": matches[-1]
    }

    # Save the output as json file
    output_path = os.path.join(json_folder, f'eval_temp{temperature}_expnum{expert_num}.json')
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    return output_data["Severity Level"]
