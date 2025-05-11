import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic facial emotion data using diffusion models')
    parser.add_argument('--output_dir', type=str, default='../emotion/synthetic', help='Output directory for synthetic images')
    parser.add_argument('--num_images', type=int, default=500, help='Number of images to generate per emotion')
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1', help='Hugging Face model ID for Stable Diffusion')
    parser.add_argument('--use_auth_token', action='store_true', help='Use Hugging Face authentication token')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def setup_output_dirs(output_dir, emotions):
    """Create output directories for each emotion."""
    for emotion in emotions:
        os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

def generate_emotion_prompt(emotion):
    """Generate diverse prompts for each emotion to create varied synthetic data."""
    base_prompts = {
        'angry': [
            "close-up portrait photo of a person with an angry facial expression, high quality, realistic",
            "detailed photograph of a person looking angry, showing anger in eyes and mouth, professional portrait",
            "clear facial portrait of someone expressing intense anger, neutral background, studio lighting",
            "high-resolution photo of an angry human face, expressive eyebrows, natural lighting",
            "professional headshot of person with furrowed brows showing anger, minimalist background"
        ],
        'disgust': [
            "close-up portrait photo of a person with a disgusted facial expression, high quality, realistic",
            "detailed photograph of a person looking disgusted, wrinkling nose, professional portrait",
            "clear facial portrait of someone expressing disgust, neutral background, studio lighting",
            "high-resolution photo of a human face showing disgust, curled upper lip, natural lighting",
            "professional headshot of person with disgusted expression, minimalist background"
        ],
        'fear': [
            "close-up portrait photo of a person with a fearful facial expression, high quality, realistic",
            "detailed photograph of a person looking scared, wide eyes, professional portrait",
            "clear facial portrait of someone expressing fear, neutral background, studio lighting",
            "high-resolution photo of a human face showing fear, raised eyebrows, natural lighting",
            "professional headshot of person with frightened expression, minimalist background"
        ],
        'happy': [
            "close-up portrait photo of a person with a happy smiling facial expression, high quality, realistic",
            "detailed photograph of a person looking happy, genuine smile, professional portrait",
            "clear facial portrait of someone expressing happiness, neutral background, studio lighting",
            "high-resolution photo of a human face showing joy, smiling eyes, natural lighting",
            "professional headshot of person with cheerful expression, minimalist background"
        ],
        'sad': [
            "close-up portrait photo of a person with a sad facial expression, high quality, realistic",
            "detailed photograph of a person looking sad, downturned mouth, professional portrait",
            "clear facial portrait of someone expressing sadness, neutral background, studio lighting",
            "high-resolution photo of a human face showing sorrow, teary eyes, natural lighting",
            "professional headshot of person with melancholic expression, minimalist background"
        ],
        'surprise': [
            "close-up portrait photo of a person with a surprised facial expression, high quality, realistic",
            "detailed photograph of a person looking surprised, open mouth, professional portrait",
            "clear facial portrait of someone expressing shock, neutral background, studio lighting",
            "high-resolution photo of a human face showing surprise, raised eyebrows, natural lighting",
            "professional headshot of person with astonished expression, minimalist background"
        ],
        'neutral': [
            "close-up portrait photo of a person with a neutral facial expression, high quality, realistic",
            "detailed photograph of a person with neutral emotion, relaxed face, professional portrait",
            "clear facial portrait of someone with emotionless expression, neutral background, studio lighting",
            "high-resolution photo of a human face showing no emotion, straight mouth, natural lighting",
            "professional headshot of person with blank expression, minimalist background"
        ]
    }
    
    # Select a random prompt template from the options
    prompt_template = random.choice(base_prompts[emotion])
    
    # Add random variations to generate diverse prompts
    age_variations = ["young", "middle-aged", "elderly", "teenage", ""]
    gender_variations = ["man", "woman", "person", ""]
    ethnicity_variations = ["Asian", "African", "European", "Middle Eastern", "Hispanic", ""]
    
    # Only apply variations sometimes to maintain diversity
    if random.random() > 0.5:
        age = random.choice(age_variations)
        gender = random.choice(gender_variations)
        ethnicity = random.choice(ethnicity_variations)
        
        # Create a subject phrase with the selected variations
        subject_parts = [part for part in [age, ethnicity, gender] if part]
        if subject_parts:
            subject = " ".join(subject_parts)
            # Replace "person" with the specific subject
            prompt_template = prompt_template.replace("person", subject)
    
    return prompt_template

def post_process_image(image):
    """Apply post-processing to make synthetic images more suitable for training."""
    # Convert to grayscale for FER2013 compatibility
    grayscale_image = image.convert('L')
    
    # Resize to 48x48 (FER2013 standard)
    resized_image = grayscale_image.resize((48, 48), Image.LANCZOS)
    
    return resized_image

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Define emotion categories
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Create output directories
    setup_output_dirs(args.output_dir, emotions)
    
    # Load Stable Diffusion model
    print(f"Loading model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_auth_token=args.use_auth_token if args.use_auth_token else None
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        print("WARNING: CUDA not available, using CPU. This will be very slow!")
    
    # Special handling for 'disgust' to address class imbalance
    images_per_emotion = {
        'angry': args.num_images,
        'disgust': args.num_images * 3,  # Generate more for the underrepresented class
        'fear': args.num_images,
        'happy': args.num_images,
        'sad': args.num_images,
        'surprise': args.num_images,
        'neutral': args.num_images
    }
    
    # Generate images for each emotion
    for emotion in emotions:
        print(f"Generating {images_per_emotion[emotion]} images for emotion: {emotion}")
        
        for i in tqdm(range(images_per_emotion[emotion])):
            # Generate a prompt for the current emotion
            prompt = generate_emotion_prompt(emotion)
            
            # Set a random seed for this specific generation
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Generate the image
            image = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            # Post-process the image
            processed_image = post_process_image(image)
            
            # Save the image
            output_path = os.path.join(args.output_dir, emotion, f"{emotion}_{i:05d}.png")
            processed_image.save(output_path)
    
    print(f"Successfully generated synthetic data in {args.output_dir}")

if __name__ == "__main__":
    main() 