{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3513e181",
   "metadata": {},
   "outputs": [],
   "source": [
    "# setup.py\n",
    "from setuptools import setup\n",
    "from torch.utils.cpp_extension import CppExtension, BuildExtension\n",
    "\n",
    "extra_compile_args = {\n",
    "    \"cxx\": [\"-O3\", \"-std=c++17\", \"-fopenmp\", \"-march=native\"]\n",
    "}\n",
    "extra_link_args = [\"-fopenmp\"]\n",
    "\n",
    "setup(\n",
    "    name=\"rosa_cpp\",\n",
    "    ext_modules=[\n",
    "        CppExtension(\n",
    "            name=\"rosa_cpp\",\n",
    "            sources=[\"rosa_kernel.cpp\"],\n",
    "            extra_compile_args=extra_compile_args,\n",
    "            extra_link_args=extra_link_args,\n",
    "        )\n",
    "    ],\n",
    "    cmdclass={\"build_ext\": BuildExtension},\n",
    ")\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
