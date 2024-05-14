#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"


int block_size = 512; // cuda thread block size
int size; // problem size


__global__ void initialize(float *data,int size) {
    // TODO: intialize the temperature distribution (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int len = size * size;
    if (i < len) {
        data[i] = wall_temp;
    }
}


__global__ void generate_fire_area(bool *fire_area,int size){
    // TODO: generate the fire area (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int len = size * size;
    if (i < len) {
        fire_area[i] = 0;
    }

    float fire1_r2 = fire_size * fire_size;
    if (i < len){
        for (int j = 0; j < size; j++){
            int a = i - size / 2;
            int b = j - size / 2;
            int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
            if (r2 < fire1_r2) fire_area[i * size + j] = 1;
        }
    }

    float fire2_r2 = (fire_size / 2) * (fire_size / 2);
    if (i < len){
        for (int j = 0; j < size; j++){
            int a = i - 1 * size / 3;
            int b = j - 1 * size / 3;
            int r2 = a * a + b * b;
            if (r2 < fire2_r2) fire_area[i * size + j] = 1;
        }
    }
}


__global__ void update(float *data, float *new_data,int size) {
    // TODO: update temperature for each point  (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size - 1){
        for (int j = 1; j < size - 1; j++){
            int idx = i * size + j;
            float up = data[idx - size];
            float down = data[idx + size];
            float left = data[idx - 1];
            float right = data[idx + 1];
            float new_val = (up + down + left + right) / 4;
            new_data[idx] = new_val;
        }
    }
    __syncthreads();
}


__global__ void maintain_wall(float *data, int size) {
    // TODO: maintain the temperature of the wall (sequential is enough)
    int len = size * size;
    for(int i=0;i<len;i++){
        if(i<size){ // up wall
            data[i] = wall_temp;
        }
        else if(i%size==0){ // left wall
            data[i] = wall_temp;
        }
        else if(i%size==size-1){ // right wall
            data[i] = wall_temp;
        }
        else if(i>len-size-1){ // down wall
            data[i] = wall_temp;
        }
    }
}


__global__ void maintain_fire(float *data, bool *fire_area, int size) {
    // TODO: maintain the temperature of the fire (in parallelized way)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int len = size * size;
    if (i<len){
        if (fire_area[i]) data[i] = fire_temp;
    }
    __syncthreads();
}


#ifdef GUI
void data2pixels(float *data, GLubyte* pixels, int size){
    // convert rawdata (large, size^2) to pixels (small, resolution^2) for faster rendering speed
    float factor_data_pixel = (float) size / resolution;
    float factor_temp_color = (float) 255 / fire_temp;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < resolution){
        for (int y = 0; y < resolution; y++){
            int idx = x * resolution + y;
            int idx_pixel = idx * 3;
            int x_raw = x * factor_data_pixel;
            int y_raw = y * factor_data_pixel;
            int idx_raw = y_raw * size + x_raw;
            float temp = data[idx_raw];
            int color =  ((int) temp / 5 * 5) * factor_temp_color;
            pixels[idx_pixel] = color;
            pixels[idx_pixel + 1] = 255 - color;
            pixels[idx_pixel + 2] = 255 - color;
        }
    }
}


void plot(GLubyte* pixels){
    // visualize temprature distribution
    #ifdef GUI
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(resolution, resolution, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glutSwapBuffers();
    #endif
}
#endif


void master() {
    float *device_data_odd;
    float *device_data_even;
    bool *device_fire_area;

    cudaMalloc(&device_data_odd, size * size * sizeof(float));
    cudaMalloc(&device_data_even, size * size * sizeof(float));
    cudaMalloc(&device_fire_area, size * size * sizeof(bool));

    #ifdef GUI
    GLubyte *pixels;
    GLubyte *host_pixels;
    host_pixels = new GLubyte[resolution * resolution * 3];
    cudaMalloc(&pixels, resolution * resolution * 3 * sizeof(GLubyte));
    #endif

    int n_block_size = size * size / block_size + 1;

    #ifdef GUI
    int n_block_resolution = resolution * resolution / block_size + 1;
    #endif 

    initialize<<<n_block_size, block_size>>>(device_data_odd,size);
    generate_fire_area<<<n_block_size, block_size>>>(device_fire_area,size);
    
    int count = 1;
    double total_time = 0;

    while (count<1000){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // TODO: modify the following lines to fit your need.
        if (count % 2 == 1) {
            update<<<n_block_size, block_size>>>(device_data_odd, device_data_even, size);
            maintain_fire<<<n_block_size, block_size>>>(device_data_even, device_fire_area, size);
            maintain_wall<<<1, 1>>>(device_data_even, size);
        } else {
            update<<<n_block_size, block_size>>>(device_data_even, device_data_odd, size);
            maintain_fire<<<n_block_size, block_size>>>(device_data_odd, device_fire_area, size);
            maintain_wall<<<1, 1>>>(device_data_odd, size);
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        total_time += this_time;
        printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        count++;
        
        #ifdef GUI
        if (count % 2 == 1) {
            data2pixels<<<n_block_resolution, block_size>>>(device_data_even, pixels, size);
        } else {
            data2pixels<<<n_block_resolution, block_size>>>(device_data_odd, pixels, size);
        }
        cudaMemcpy(host_pixels, pixels, resolution * resolution * 3 * sizeof(GLubyte), cudaMemcpyDeviceToHost);
        plot(host_pixels);
        #endif
    }


    cudaFree(device_data_odd);
    cudaFree(device_data_even);
    cudaFree(device_fire_area);

    #ifdef GUI
    cudaFree(pixels);
    delete[] host_pixels;
    #endif
}


int main(int argc, char *argv[]){
    size = atoi(argv[1]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(resolution, resolution);
    glutCreateWindow("Heat Distribution Simulation Sequential Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
    #endif

    master();

    printf("Student ID: 120090414\n"); // replace it with your student id
    printf("Name: Xiang Fei\n"); // replace it with your name
    printf("Assignment 4: Heat Distribution CUDA Implementation\n");

    return 0;
}
