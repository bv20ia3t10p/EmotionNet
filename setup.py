from setuptools import setup, find_packages

setup(
    name="emotion_net",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "tqdm>=4.50.0",
    ],
    author="EmotionNet Team",
    description="A deep learning package for emotion recognition",
    python_requires=">=3.7",
) 