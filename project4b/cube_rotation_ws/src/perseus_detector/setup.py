import os
from glob import glob
from warnings import simplefilter

from setuptools import SetuptoolsDeprecationWarning, setup

simplefilter("ignore", category=SetuptoolsDeprecationWarning)

package_name = "perseus_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name, f"{package_name}.detector"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="alberthli",
    maintainer_email="alberthli@caltech.edu",
    description="Perseus detector model package for ROS2.",
    license="MIT",
    tests_require=["pytest"],
)
