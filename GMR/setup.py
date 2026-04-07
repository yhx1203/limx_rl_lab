from pathlib import Path

from setuptools import setup, find_packages

description = "General Motion Retargeting (GMR) for Humanoid Robots"
readme_path = Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else description

setup(
  name = 'general_motion_retargeting',
  packages = find_packages(),
  author="Yanjie Ze",
  author_email="lastyanjieze@gmail.com",
  description=description,
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/YanjieZe/GMR",
  license="MIT",
  version="0.2.0",
  install_requires=[
    "loop_rate_limiters",
    "mink",
    "mujoco",
    "numpy==1.26.0",
    "scipy",
    "qpsolvers[proxqp]",
    "rich",
    "tqdm",
    "opencv-python==4.11.0.86",
    "natsort",
    "psutil",
    "smplx @ git+https://github.com/vchoutas/smplx",
    "protobuf",
    "redis[hiredis]",
    "imageio[ffmpeg]",
  ],
  python_requires='>=3.10',
)
