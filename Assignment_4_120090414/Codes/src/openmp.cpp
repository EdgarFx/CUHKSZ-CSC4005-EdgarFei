#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif


#include "./headers/physics.h"


int size; // problem size
int n_omp_threads;


void initialize(float *data) {
    // intialize the temperature distribution
    int len = size * size;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        data[i] = wall_temp;
    }
}


void generate_fire_area(bool *fire_area){
    // generate the fire area
    int len = size * size;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        fire_area[i] = 0;
    }

    float fire1_r2 = fire_size * fire_size;
    int j1;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for private(j1)
    for (int i = 0; i < size; i++){
        for (j1 = 0; j1 < size; j1++){
            int a = i - size / 2;
            int b = j1 - size / 2;
            int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
            if (r2 < fire1_r2) fire_area[i * size + j1] = 1;
        }
    }

    float fire2_r2 = (fire_size / 2) * (fire_size / 2);
    int j2;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for private(j2)
    for (int i = 0; i < size; i++){
        for (j2 = 0; j2 < size; j2++){
            int a = i - 1 * size / 3;
            int b = j2 - 1 * size / 3;
            int r2 = a * a + b * b;
            if (r2 < fire2_r2) fire_area[i * size + j2] = 1;
        }
    }
}


void update(float *data, float *new_data) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    int j;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for private(j)
    for (int i = 1; i < size - 1; i++){
        for (j = 1; j < size - 1; j++){
            int idx = i * size + j;
            float up = data[idx - size];
            float down = data[idx + size];
            float left = data[idx - 1];
            float right = data[idx + 1];
            float new_val = (up + down + left + right) / 4;
            new_data[idx] = new_val;
        }
    }
}


void maintain_fire(float *data, bool* fire_area) {
    // maintain the temperature of fire
    int len = size * size;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for
    for (int i = 0; i < len; i++){
        if (fire_area[i]) data[i] = fire_temp;
    }
}


void maintain_wall(float *data) {
    // TODO: maintain the temperature of the wall
    int len = size * size;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for
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


#ifdef GUI
void data2pixels(float *data, GLubyte* pixels){
    // convert rawdata (large, size^2) to pixels (small, resolution^2) for faster rendering speed
    float factor_data_pixel = (float) size / resolution;
    float factor_temp_color = (float) 255 / fire_temp;
    int y;
    #pragma omp parallel for private(y)
    for (int x = 0; x < resolution; x++){
        for (y = 0; y < resolution; y++){
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


void master(){
    float *data_odd;
    float *data_even;
    bool *fire_area;

    #ifdef GUI
    GLubyte* pixels = new GLubyte[resolution * resolution * 3];
    #endif
    
    data_odd = new float[size * size];
    data_even = new float[size * size];

    fire_area = new bool[size * size];

    generate_fire_area(fire_area);
    initialize(data_odd);

    int count = 1;
    double total_time = 0;

    while (true) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        if (count % 2 == 1) {
            update(data_odd, data_even);
            maintain_fire(data_even, fire_area);
            maintain_wall(data_even);
        } else {
            update(data_even, data_odd);
            maintain_fire(data_odd, fire_area);
            maintain_wall(data_odd);
        }
        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        total_time += this_time;
        printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        count++;


        #ifdef GUI
        if (count % 2 == 1) {
            data2pixels(data_even, pixels);
        } else {
            data2pixels(data_odd, pixels);
        }
        plot(pixels);
        #endif
        
    }

    delete[] data_odd;
    delete[] data_even;
    delete[] fire_area;

    #ifdef GUI
    delete[] pixels;
    #endif
  
}


int main(int argc, char* argv[]) {
    size = atoi(argv[1]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(resolution, resolution);
    glutCreateWindow("Heat Distribution Simulation OpenMP Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
    #endif

    master();

    printf("Student ID: 120090414\n"); // replace it with your student id
    printf("Name: Xiang Fei\n"); // replace it with your name
    printf("Assignment 4: Heat Distribution OpenMP Implementation\n");

    return 0;
}