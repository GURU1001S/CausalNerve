from setuptools import setup, find_packages

setup(
    name="causalnerve-observe",
    version="1.0.4",
    packages=find_packages(),
    install_requires=[
        "causalnerve>=1.0.4",
        "plotly>=5.0",
        "dash>=2.0",
        "gradio>=4.0",
    ],
    description="CausalNerve Observatory and visualization package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
