from setuptools import setup, find_packages

setup(
    name='huaweicloud-python-sdk-sis',
    version='1.8.1',
	author='Huaweicloud SIS',
    packages=find_packages(),
    zip_safe=False,
    description='sis python sdk',
    long_description='sis python sdk',
	install_requires=['websocket-client', 'requests'],
    license='Apache-2.0',
    keywords=('sis', 'sdk', 'python'),
    platforms='Independant'
)