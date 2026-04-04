import os
from glob import glob
from warnings import simplefilter

from setuptools import SetuptoolsDeprecationWarning, setup

simplefilter("ignore", category=SetuptoolsDeprecationWarning)

package_name = "cro_estimation"
assets_files = [path for path in glob(os.path.join("assets", "**"), recursive=True) if os.path.isfile(path)]

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "ckpts"), glob(os.path.join("ckpts", "*.pth"))),
        (os.path.join("share", package_name, "assets"), assets_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yuvansharma",
    maintainer_email="yuvan@berkeley.edu",
    description="Minimal ROS2 cube pose estimation package using a single camera, adapted from DROP paper from Caltech.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"cro_estimator = {package_name}.cro_estimator:main",
        ],
    },
)
