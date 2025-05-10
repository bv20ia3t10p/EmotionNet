import albumentations as A  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
from config import *

# Get phase-dependent augmentation params


def get_phase_augmentation_params(phase=5):
    """Returns augmentation parameters based on training phase (1-5).

    Args:
        phase: The training phase (1=minimal, 5=maximum)

    Returns:
        Dictionary of augmentation parameters
    """
    # Base parameters (phase 1 - minimal)
    base_params = {
        "scale": (0.9, 1.1),                 # Random scaling
        "rotate_limit": 15,                  # Rotation range
        "brightness_contrast_limit": 0.15,   # Brightness/contrast adjustment
        "hue_shift": 10,                     # Hue shift
        "sat_shift": 20,                     # Saturation shift
        "val_shift": 15,                     # Value shift
        "noise_prob": 0.15,                  # Noise probability (increased)
        "blur_prob": 0.15,                   # Blur probability (increased)
        "distortion_prob": 0.1,              # Distortion probability
        "cutout_prob": 0.1,                  # Cutout probability (increased)
        # Random gamma probability (increased)
        "random_gamma_prob": 0.1,
        # Random shadow probability (increased)
        "random_shadow_prob": 0.1,
        # Perspective transform probability (increased)
        "perspective_prob": 0.1,
        "elastic_prob": 0.0,                 # Elastic transform probability
        "grid_distortion_prob": 0.0,         # Grid distortion probability
        "optical_distortion_prob": 0.0,      # Optical distortion probability
        "channel_shuffle_prob": 0.0,         # Channel shuffle probability
        "fancy_pca_prob": 0.0,               # Fancy PCA probability
        "random_fog_prob": 0.0,              # Random fog probability
        # Spatial transforms probability (increased)
        "spatial_prob": 0.3,
        "face_aware_prob": 0.1,              # Face aware operations (NEW)
        # Probability to crop around face regions (NEW)
        "crop_face_prob": 0.15,
        # Emotion-specific augmentations (NEW)
        "emotion_specific_prob": 0.05,
    }

    # Phase 2 - moderate
    if phase >= 2:
        base_params.update({
            "scale": (0.8, 1.2),
            "rotate_limit": 20,
            "brightness_contrast_limit": 0.2,
            "hue_shift": 15,
            "sat_shift": 30,
            "val_shift": 20,
            "noise_prob": 0.2,
            "blur_prob": 0.2,
            "distortion_prob": 0.25,
            "cutout_prob": 0.2,
            "random_gamma_prob": 0.15,
            "random_shadow_prob": 0.15,
            "perspective_prob": 0.15,
            "spatial_prob": 0.5,
            "face_aware_prob": 0.15,
            "crop_face_prob": 0.2,
            "emotion_specific_prob": 0.1,
        })

    # Phase 3 - strong
    if phase >= 3:
        base_params.update({
            "scale": (0.75, 1.25),
            "rotate_limit": 25,
            "brightness_contrast_limit": 0.25,
            "hue_shift": 20,
            "sat_shift": 35,
            "val_shift": 25,
            "noise_prob": 0.25,
            "blur_prob": 0.25,
            "distortion_prob": 0.35,
            "cutout_prob": 0.3,
            "random_gamma_prob": 0.2,
            "random_shadow_prob": 0.2,
            "perspective_prob": 0.2,
            "elastic_prob": 0.15,
            "grid_distortion_prob": 0.15,
            "channel_shuffle_prob": 0.1,
            "spatial_prob": 0.7,
            "face_aware_prob": 0.2,
            "crop_face_prob": 0.25,
            "emotion_specific_prob": 0.15,
        })

    # Phase 4 - very strong
    if phase >= 4:
        base_params.update({
            "scale": (0.7, 1.3),
            "rotate_limit": 30,
            "brightness_contrast_limit": 0.3,
            "hue_shift": 25,
            "sat_shift": 45,
            "val_shift": 30,
            "noise_prob": 0.35,
            "blur_prob": 0.3,
            "distortion_prob": 0.45,
            "cutout_prob": 0.35,
            "random_gamma_prob": 0.25,
            "random_shadow_prob": 0.25,
            "perspective_prob": 0.25,
            "elastic_prob": 0.2,
            "grid_distortion_prob": 0.2,
            "optical_distortion_prob": 0.15,
            "channel_shuffle_prob": 0.15,
            "fancy_pca_prob": 0.1,
            "random_fog_prob": 0.1,
            "spatial_prob": 0.8,
            "face_aware_prob": 0.25,
            "crop_face_prob": 0.3,
            "emotion_specific_prob": 0.2,
        })

    # Phase 5 - maximum (for final stages of training)
    if phase >= 5:
        base_params.update({
            "scale": (0.65, 1.35),
            "rotate_limit": 35,
            "brightness_contrast_limit": 0.35,
            "hue_shift": 30,
            "sat_shift": 50,
            "val_shift": 35,
            "noise_prob": 0.4,
            "blur_prob": 0.35,
            "distortion_prob": 0.5,
            "cutout_prob": 0.45,
            "random_gamma_prob": 0.3,
            "random_shadow_prob": 0.3,
            "perspective_prob": 0.3,
            "elastic_prob": 0.25,
            "grid_distortion_prob": 0.25,
            "optical_distortion_prob": 0.2,
            "channel_shuffle_prob": 0.2,
            "fancy_pca_prob": 0.15,
            "random_fog_prob": 0.15,
            "spatial_prob": 0.85,
            "face_aware_prob": 0.3,
            "crop_face_prob": 0.35,
            "emotion_specific_prob": 0.25,
        })

    return base_params


# Create advanced train transforms with richer augmentation strategies
def get_train_transform(phase=5):
    """Get training transforms with phase-dependent augmentation strength.

    Args:
        phase: The training phase (1-5), with higher phases having stronger augmentation

    Returns:
        Albumentations transform pipeline
    """
    params = get_phase_augmentation_params(phase)

    return A.Compose([
        # Spatial transforms - enhanced with face-aware transforms
        A.OneOf([
            A.RandomResizedCrop(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                scale=(params["scale"][0], params["scale"][1]),
                ratio=(0.75, 1.33),
                p=1.0
            ),
            A.Sequential([
                A.Resize(int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1)),
                A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE)
            ], p=1.0),
            # Add more aggressive crops for stronger regularization
            A.Sequential([
                A.Resize(int(IMAGE_SIZE * 1.25), int(IMAGE_SIZE * 1.25)),
                A.RandomSizedCrop(min_max_height=(int(IMAGE_SIZE * 0.8), IMAGE_SIZE),
                                  height=IMAGE_SIZE, width=IMAGE_SIZE,
                                  w2h_ratio=1.0)
            ], p=0.3 if phase >= 2 else 0.0),
            # NEW: Face-aware crops (simulated since Albumentations doesn't have this directly)
            A.Sequential([
                A.Resize(int(IMAGE_SIZE * 1.5), int(IMAGE_SIZE * 1.5)),
                A.RandomSizedCrop(
                    min_max_height=(int(IMAGE_SIZE * 0.9), IMAGE_SIZE),
                    height=IMAGE_SIZE, width=IMAGE_SIZE,
                    w2h_ratio=0.9  # Face-region likely ratio
                )
            ], p=params["crop_face_prob"]),
        ], p=params["spatial_prob"]),

        # Basic augmentations - always applied with varying probabilities
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05 if phase == 1 else 0.1),  # Reduced for face data
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=params["rotate_limit"],
            p=0.7
        ),

        # Color transforms - medium probability - stronger for higher phases
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=params["brightness_contrast_limit"],
                contrast_limit=params["brightness_contrast_limit"]
            ),
            A.HueSaturationValue(
                hue_shift_limit=params["hue_shift"],
                sat_shift_limit=params["sat_shift"],
                val_shift_limit=params["val_shift"]
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            A.CLAHE(clip_limit=4.0, p=0.5),
            # Add more aggressive color transforms for phase 2+
            A.RGBShift(r_shift_limit=30, g_shift_limit=30,
                       b_shift_limit=30, p=1.0)
        ], p=0.8),

        # Emotion-specific augmentations (more targeted for face data)
        A.OneOf([
            # Mimic lighting changes that affect emotion perception
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),  # Focus on lower part of face
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5
            ),
            # Selective blurring for emotion ambiguity
            A.Blur(blur_limit=3),
            # Slight sharpening to enhance expressions
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
        ], p=params["emotion_specific_prob"] * 2 if phase >= 3 else params["emotion_specific_prob"]),

        # Noise transforms - increased probability for higher phases
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            # Add stronger noise for phase 3+
            A.ImageCompression(
                quality_lower=65, quality_upper=100, p=1.0 if phase >= 3 else 0.0)
        ], p=params["noise_prob"] * (1.0 if phase == 1 else 1.5)),

        # Blur transforms - increased for higher phases
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GlassBlur(sigma=0.7, max_delta=2),
            # Add stronger blur for phase 3+
            A.ZoomBlur(max_factor=1.3, p=1.0 if phase >= 3 else 0.0)
        ], p=params["blur_prob"]),

        # Distortion transforms - increased for higher phases
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1),
            A.Perspective(scale=(0.05, 0.1)),
            # Add more aggressive distortions for phase 3+
            A.GridDropout(ratio=0.3, holes_number_x=5,
                          holes_number_y=5, p=1.0 if phase >= 3 else 0.0)
        ], p=params["distortion_prob"]),

        # Advanced augmentations - phase-dependent probability
        A.CoarseDropout(
            max_holes=12 if phase >= 3 else 8,
            max_height=IMAGE_SIZE//6 if phase >= 3 else IMAGE_SIZE//8,
            max_width=IMAGE_SIZE//6 if phase >= 3 else IMAGE_SIZE//8,
            min_holes=1,
            min_height=IMAGE_SIZE//16,
            min_width=IMAGE_SIZE//16,
            p=params["cutout_prob"]
        ),

        # Stronger random erasing for phase 2+
        # Occasionally convert to grayscale
        A.ToGray(p=0.1 if phase >= 2 else 0.0),

        A.RandomGamma(gamma_limit=(80, 120), p=params["random_gamma_prob"]),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=3 if phase >= 3 else 2,
            p=params["random_shadow_prob"]
        ),
        A.PiecewiseAffine(scale=(0.01, 0.03), p=params["elastic_prob"]),
        A.ChannelShuffle(p=params["channel_shuffle_prob"]),

        # Weather augmentations - stronger for higher phases but reduced for emotion data
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1,
                        fog_coef_upper=0.2, alpha_coef=0.1),
            A.RandomRain(drop_length=10 if phase >= 3 else 5,
                         blur_value=3, brightness_coefficient=0.9),
            A.RandomSnow(snow_point_lower=0.1,
                         snow_point_upper=0.2, brightness_coeff=0.9),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),
                             angle_lower=0, angle_upper=1),
        ], p=params["random_fog_prob"] * 0.7),  # Reduced for emotion data

        # Final normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# Robust validation transforms with TTA-friendly normalization
def get_val_transform():
    """Get validation transforms.

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=int(IMAGE_SIZE * 1.1), width=int(IMAGE_SIZE * 1.1)),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# Enhanced test-time augmentation transforms for more robust inference
def get_tta_transforms(num_augments=16):  # Increased from default 5
    """Get a list of test-time augmentation transforms.

    Args:
        num_augments: Number of different augmentations to create

    Returns:
        List of Albumentations transform pipelines
    """
    base_transforms = [
        # Original center crop - always included
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1),
                     width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),

        # Horizontal flip - always included
        A.Compose([
            A.Resize(height=int(IMAGE_SIZE * 1.1),
                     width=int(IMAGE_SIZE * 1.1)),
            A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

    # Add more transforms if requested
    if num_augments > 2:
        # Add brightness variations
        for brightness in [-0.15, -0.05, 0.05, 0.15]:
            base_transforms.append(
                A.Compose([
                    A.Resize(height=int(IMAGE_SIZE * 1.1),
                             width=int(IMAGE_SIZE * 1.1)),
                    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
                    A.RandomBrightnessContrast(brightness_limit=(
                        brightness, brightness), contrast_limit=0, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            )

    if num_augments > 6:
        # Add slight rotations
        for angle in [-10, -5, 5, 10]:
            base_transforms.append(
                A.Compose([
                    A.Resize(height=int(IMAGE_SIZE * 1.1),
                             width=int(IMAGE_SIZE * 1.1)),
                    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
                    A.Rotate(limit=(angle, angle), p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            )

    if num_augments > 10:
        # Add slight shifts
        for shift_x, shift_y in [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]:
            base_transforms.append(
                A.Compose([
                    A.Resize(height=int(IMAGE_SIZE * 1.1),
                             width=int(IMAGE_SIZE * 1.1)),
                    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
                    A.ShiftScaleRotate(shift_limit_x=(shift_x, shift_x), shift_limit_y=(shift_y, shift_y),
                                       scale_limit=0, rotate_limit=0, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            )

    if num_augments > 14:
        # Add contrast and combination transforms
        for contrast in [-0.1, 0.1]:
            base_transforms.append(
                A.Compose([
                    A.Resize(height=int(IMAGE_SIZE * 1.1),
                             width=int(IMAGE_SIZE * 1.1)),
                    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
                    A.RandomBrightnessContrast(
                        brightness_limit=0, contrast_limit=(contrast, contrast), p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            )

        # Add two zoom levels
        for scale in [0.9, 1.1]:
            base_transforms.append(
                A.Compose([
                    A.Resize(height=int(IMAGE_SIZE * 1.1),
                             width=int(IMAGE_SIZE * 1.1)),
                    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=(
                        scale-1, scale-1), rotate_limit=0, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            )

    # Return the full list or truncate to requested number
    return base_transforms[:num_augments]


# Create more advanced transforms for integrating with existing code
def create_transforms(phase=5, tta_augments=5):
    """Create transform functions for training, validation, and TTA.

    Args:
        phase: The training phase (1-5)
        tta_augments: Number of TTA augmentations

    Returns:
        Dictionary with train_transform, val_transform, and tta_transforms
    """
    return {
        "train_transform": get_train_transform(phase=phase),
        "val_transform": get_val_transform(),
        "tta_transforms": get_tta_transforms(num_augments=tta_augments)
    }
