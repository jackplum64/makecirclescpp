# Circle Packing Image Generation Project

## Overview
This project generates images of circles with specified statistical properties using a circle packing algorithm. The circles are generated in two groups with different properties, and the images are saved both in their original and blurred forms. The project uses OpenCV for image processing and rendering.

## Dependencies
- [OpenCV 4.x](https://opencv.org/get-started/)

## Project Structure
- `main.cpp`: The main program file containing the circle generation, packing, and image saving logic.
- `config.ini`: Configuration file specifying the properties for circle generation and image dimensions.
- `CMakeLists.txt`: CMake build script.
- `README.md`: Project documentation.

## Usage

### Compilation
1. Clone the repository and navigate to the root directory:
   ```sh
   git clone https://github.com/jackplum64/makecirclescpp.git
   cd makecirclescpp
   ```
2. Run `make` to build the project:
   ```sh
   make
   ```

### Running the Program
1. Ensure you have the configuration file (`config.ini`) in the same directory as the executable or specify the path.
2. Run the executable with the configuration file:
   ```sh
   ./makecircles -config config.ini
   ```
3. The program will generate the specified number of images and save them in the `output` directory within the parent directory of the executable.

## Configuration File (`config.ini`)
The configuration file contains parameters for the image dimensions, the number of outputs, and the properties of the two groups of circles. Below is an example configuration:

```ini
# GENERAL #
width=1000 # int - image width
height=1000 # int - image height
numOutputs=12 # int - number of generated images


# GROUP 1 #
group1_mean=73.0 # float - target radius mean
group1_mean_delta=0.5 # float - max allowed difference between target mean and actual mean
group1_std_dev=32.0 # float - target radius std_dev
group1_std_dev_delta=0.5 # float - max allowed difference between target std_dev and actual std_dev
group1_count=36 # int - number of circles

group1_color_r=0 # int - 0 to 255
group1_color_g=0 # int - 0 to 255
group1_color_b=255 # int - 0 to 255


# GROUP 2 #
group2_mean=10.0 # float - target radius mean
group2_mean_delta=0.5 # float - max allowed difference between target mean and actual mean
group2_std_dev=6.0 # float - target radius std_dev
group2_std_dev_delta=0.5 # float - max allowed difference between target std_dev and actual std_dev
group2_count=86 # int - number of circles

group2_color_r=255 # int - 0 to 255
group2_color_g=0 # int - 0 to 255
group2_color_b=0 # int - 0 to 255
```

### Circle Packing Constraints
- **Group 1**: Circles will not overlap with themselves.
- **Group 2**: Circles will not overlap with themselves and will not overlap fully with Group 1 circles, but can overlap partially.

## Output
The generated images will be saved in the `output` directory within the parent directory of the executable. Each image will be saved twice: once in its original form and once with a Gaussian blur applied.

### Example
For a configuration with `numOutputs=1`, the following files will be generated:
- `output/output_g1_mean_73.00_std_32.00_g1_count_36_g2_mean_10.00_std_6.00_g2_count_86/circles_0.png`: Original image.
- `output/output_g1_mean_73.00_std_32.00_g1_count_36_g2_mean_10.00_std_6.00_g2_count_86/circles_blurred_0.png`: Blurred image.
Note, your output directory name will change with the inputs in config.ini
