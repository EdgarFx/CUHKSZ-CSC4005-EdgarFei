#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif


#include "./headers/physics.h"

int size; // problem size
int thread_num; // thread number
int batch_size;
int current_iteration;
int max_iteration;
pthread_mutex_t mutex;
pthread_barrier_t barrier;


void initialize(float *data) {
    // intialize the temperature distribution
    int len = size * size;
    for (int i = 0; i < len; i++) {
        data[i] = wall_temp;
    }
}


void generate_fire_area(bool *fire_area){
    // generate the fire area
    int len = size * size;
    for (int i = 0; i < len; i++) {
        fire_area[i] = 0;
    }

    float fire1_r2 = fire_size * fire_size;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            int a = i - size / 2;
            int b = j - size / 2;
            int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
            if (r2 < fire1_r2) fire_area[i * size + j] = 1;
        }
    }

    float fire2_r2 = (fire_size / 2) * (fire_size / 2);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            int a = i - 1 * size / 3;
            int b = j - 1 * size / 3;
            int r2 = a * a + b * b;
            if (r2 < fire2_r2) fire_area[i * size + j] = 1;
        }
    }
}


void update(float *data, float *new_data) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    int local_iteration = current_iteration+1;
    current_iteration++;
    int len = size * size;
    while(true) {
        int start = local_iteration*batch_size;
        int end = local_iteration*batch_size + batch_size;
        for (int i = start; i < end; i++){
            if ((i<size)||(i%size==0)||(i%size==size-1)||(i>len-size-1)){
                continue;
            }
            float up = data[i - size];
            float down = data[i + size];
            float left = data[i - 1];
            float right = data[i + 1];
            float new_val = (up + down + left + right) / 4;
            new_data[i] = new_val;
        }
        pthread_mutex_lock(&mutex);
        local_iteration = current_iteration + 1;
        current_iteration++;
        pthread_mutex_unlock(&mutex);
        if (local_iteration > max_iteration-1){
            break;
        } 
    }
}


void maintain_fire(float *data, bool* fire_area) {
    // maintain the temperature of fire
    int local_iteration = current_iteration+1;
    current_iteration++;
    while(true) {
        for (int i = 0; i < batch_size; i++){
            if (fire_area[i + local_iteration*batch_size]){
                data[i + local_iteration*batch_size] = fire_temp;
            }
        }
        pthread_mutex_lock(&mutex);
        local_iteration = current_iteration + 1;
        current_iteration++;
        pthread_mutex_unlock(&mutex);
        if (local_iteration > max_iteration-1){
            break;
        } 
    }
}


void maintain_wall(float *data) {
    // TODO: maintain the temperature of the wall
    int local_iteration = current_iteration+1;
    current_iteration++;
    while(true) {
        int len = size * size;
        for(int i=0;i<batch_size;i++){
            int j = i + local_iteration*batch_size;
            if(j<size){ // up wall
                data[j] = wall_temp;
            }
            else if(j%size==0){ // left wall
                data[j] = wall_temp;
            }
            else if(j%size==size-1){ // right wall
                data[j] = wall_temp;
            }
            else if(j>len-size-1){ // down wall
                data[j] = wall_temp;
            }
        }
        pthread_mutex_lock(&mutex);
        local_iteration = current_iteration + 1;
        current_iteration++;
        pthread_mutex_unlock(&mutex);
        if (local_iteration > max_iteration-1){
            break;
        } 
    }
}


#ifdef GUI
void data2pixels(float *data, GLubyte* pixels){
    // convert rawdata (large, size^2) to pixels (small, resolution^2) for faster rendering speed
    float factor_data_pixel = (float) size / resolution;
    float factor_temp_color = (float) 255 / fire_temp;
    for (int x = 0; x < resolution; x++){
        for (int y = 0; y < resolution; y++){
            int j = x * resolution + y;
            int j_pixel = j * 3;
            int x_raw = x * factor_data_pixel;
            int y_raw = y * factor_data_pixel;
            int j_raw = y_raw * size + x_raw;
            float temp = data[j_raw];
            int color =  ((int) temp / 5 * 5) * factor_temp_color;
            pixels[j_pixel] = color;
            pixels[j_pixel + 1] = 255 - color;
            pixels[j_pixel + 2] = 255 - color;
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


typedef struct {
    float *data_odd;
    float *data_even;
    bool *fire_area;
    int count;
    int thread_id;
} Args;


void *worker(void *args) {
    Args* arg = (Args*)args;
    if (arg->count%2==1) {
        update(arg->data_odd, arg->data_even);
        pthread_barrier_wait(&barrier);
        current_iteration = 0;
        maintain_fire(arg->data_even, arg->fire_area);
        pthread_barrier_wait(&barrier);
        current_iteration = 0;
        maintain_wall(arg->data_even);
    } else {
        update(arg->data_even, arg->data_odd);
        pthread_barrier_wait(&barrier);
        current_iteration = 0;
        maintain_fire(arg->data_odd, arg->fire_area);
        pthread_barrier_wait(&barrier);
        current_iteration = 0;
        maintain_wall(arg->data_odd);
    }
}


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
        pthread_t threads[thread_num];
        Args args[thread_num];
        for (int i=0; i<thread_num; i++) {
            args[i].data_even = data_even;
            args[i].data_odd = data_odd;
            args[i].fire_area = fire_area;
            args[i].thread_id = i;
            args[i].count = count;
        }
        batch_size = 100;
        current_iteration = 0;
        max_iteration = (size*size) / batch_size;
        pthread_barrier_init(&barrier, NULL, thread_num);
        pthread_mutex_init(&mutex, NULL);
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        for (int thread = 0; thread < thread_num; thread++) pthread_create(&threads[thread], NULL, worker, &args[thread]);
        for (int thread = 0; thread < thread_num; thread++) pthread_join(threads[thread], NULL);
        
        
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

    pthread_mutex_destroy(&mutex);
    pthread_barrier_destroy(&barrier);

    delete[] data_odd;
    delete[] data_even;
    delete[] fire_area;

    #ifdef GUI
    delete[] pixels;
    #endif
  
}


int main(int argc, char* argv[]) {
    size = atoi(argv[1]);
    thread_num = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(resolution, resolution);
    glutCreateWindow("Heat Distribution Simulation Pthread Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
    #endif

    master();

    printf("Student ID: 120090414\n"); // replace it with your student id
    printf("Name: Xiang Fei\n"); // replace it with your name
    printf("Assignment 4: Heat Distribution Pthread Implementation\n");

    return 0;
}