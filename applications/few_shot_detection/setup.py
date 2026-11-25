from setuptools import setup, find_packages
setup(name="few-shot-detection", version="0.1.0",
      packages=["few_shot_detection", *find_packages("src/dinov3_cases")],
      package_dir={"few_shot_detection": "src/few_shot_detection", "dinov3_cases": "src/dinov3_cases/dinov3_cases"},
      package_data={"dinov3_cases.api": ["templates/*.html", "static/*.js", "static/*.css"]},
      install_requires=["robotic-inference-sdk", "Pillow>=10"],
      entry_points={"console_scripts": ["few-shot-detection=few_shot_detection.__main__:main"]})
