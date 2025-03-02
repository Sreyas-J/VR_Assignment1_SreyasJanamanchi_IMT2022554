
# ASSIGNMENT-1

**Part-1**

This project presents a computer vision-based approach to detecting, segmenting, and counting Indian coins from an image containing scattered coins. The implementation utilizes OpenCV and NumPy to process the image and apply various image processing techniques for accurate detection and segmentation of individual coins.

**Part-2**

The objective of this task is to create a stitched panorama from multiple overlapping images using keypoint detection and homography transformation.


## Documentation

[Documentation](https://github.com/Sreyas-J/VR_Assignment1_SreyasJanamanchi_IMT2022554/blob/main/IMT2022554.pdf)

```bash
VRassignment1/
├── part1/
│   ├── part1.py
│   ├── outline/
|   ├── contours/
|   ├── segmentedCoins/
|   ├── coins.jpeg
|   ├── coins2.jpeg
├── part2/
│   ├── in1/
│   ├── in2/
|   ├── out/
|   ├── part2.py
├── README.md
├── requirements.txt
├── environment.yml
├── VR Assignment 1.pdf
```

- part1/outline/: Contains the image with the edges being outlined.
- part1/contours/: Contains the image with only the coins visible.
- part1/segmentedCoins/: Contains images of segemted coin. Each coin has its own image.
- part1/coins.jpeg: Its an example input.
- part1/coins2.jpeg: Its an example input.
- part1/part1.py: This is the python script for part1.
- part2/in1/: It contains a set of inputs (3 images).
- part2/in2/: It contains a set of inputs (4 images).
- part2/out/: It contains images which display the matches and the final panaroma.
- part2/part2.py: This is the python script for part2.
- requirements.txt: Contains the packages available in the development conda environment.
- IMT2022554.pdf: This is the report/documentation of the assignment.

## Run Locally

Cloning the Github repository using SSH

```bash
  git clone git@github.com:Sreyas-J/VR_Assignment1_SreyasJanamanchi_IMT2022554.git
```
or

Cloning the Github repository using HTTPS

```bash
    git clone https://github.com/Sreyas-J/VR_Assignment1_SreyasJanamanchi_IMT2022554.git
```

Installing the required packages in a conda environment

```bash
    cd VR_Assignment1_SreyasJanamanchi_IMT2022554
    conda create --name <env_name> --file requirements.txt
```

Running part1

The input can be changed by updating the following variable: input_image

```bash
    cd part1
    python3 part1.py
```

Running part2

The inputs can be changed by updating the following variables:-
- input_folder
- NumberOfInputs

```bash
    cd part2
    python3 part2.py
```

