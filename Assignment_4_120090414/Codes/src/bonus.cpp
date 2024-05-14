#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>
#include <omp.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"


int size; // problem size


int my_rank;
int world_size;

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


void update(float *data, float *new_data, float* down_bound,float* up_bound) {
    // TODO: update the temperature of each point, and store the result in `new_data` to avoid data racing
    int len = size / world_size;
    if (my_rank!=0){
        for (int i=0;i<size;i++){
            up_bound[i] = data[i];
        }
        MPI_Send(up_bound,size,MPI_FLOAT,my_rank-1,my_rank,MPI_COMM_WORLD);
    }
    if (my_rank!=world_size-1){
        MPI_Recv(up_bound,size,MPI_FLOAT,my_rank+1,my_rank+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    if (my_rank!=world_size-1){
        for (int i=0;i<size;i++){
            down_bound[i] = data[(len-1)*size+i];
        }
        MPI_Send(down_bound,size,MPI_FLOAT,my_rank+1,my_rank,MPI_COMM_WORLD);
    }
    if (my_rank!=0){
        MPI_Recv(down_bound,size,MPI_FLOAT,my_rank-1,my_rank-1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    int j;
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for private(j)
    for (int i = 0; i < len; i++){
        for (j = 1; j < size - 1; j++){
            if ((my_rank*len+i!=0)&&(my_rank*len+i!=size-1)){
                int k = i*size+j;
                float up;
                float down;
                if (i!=0){
                    up = data[k - size];
                }
                else{
                    up = down_bound[j];
                }
                if(i!=len-1){
                    down = data[k + size];
                }
                else{
                    down = up_bound[j];
                }
                float left = data[k - 1];
                float right = data[k + 1];
                float new_val = (up + down + left + right) / 4;
                new_data[k] = new_val;
            }
        }
    }
}


void maintain_fire(float *data, bool* fire_area) {
    // TODO: maintain the temperature of fire
    int len = (size*size / world_size);
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for
    for (int i = 0; i < len; i++){
        if (fire_area[i]) data[i] = fire_temp;
    }
}


void maintain_wall(float *data) {
    // TODO: maintain the temperature of the wall
    int len = (size*size / world_size);
    omp_set_num_threads(n_omp_threads);
    #pragma omp parallel for
    for(int i=my_rank*len;i<len;i++){
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
    for (int x = 0; x < resolution; x++){
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
    // TODO: MPI routine (one possible solution, you can use another partition method)
    float* data_odd = new float[size * size];
    float* data_even = new float[size * size];
    bool* fire_area = new bool[size * size];

    float* slave_odd = new float[size* size / world_size];
    float* slave_even = new float[size* size / world_size];
    bool* slave_fire_area = new bool[size* size / world_size];

    float* down_bound = new float[size];
    float* up_bound = new float[size];

    int count = 1;
    double total_time,this_time;
    std::chrono::high_resolution_clock::time_point t1,t2;
    #ifdef GUI
    GLubyte* pixels;
    pixels = new GLubyte[resolution * resolution * 3];
    #endif

    if (my_rank == 0){
        initialize(data_odd);
        generate_fire_area(fire_area);
        total_time = 0;
    }

    MPI_Scatter(fire_area,(size*size/world_size),MPI_BYTE,slave_fire_area,(size*size/world_size),MPI_BYTE,0,MPI_COMM_WORLD);  
    while (true){        
        MPI_Scatter(data_odd,(size*size/world_size),MPI_FLOAT,slave_odd,(size*size/world_size),MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Scatter(data_even,(size*size/world_size),MPI_FLOAT,slave_even,(size*size/world_size),MPI_FLOAT,0,MPI_COMM_WORLD);
        if (my_rank == 0){
            t1 = std::chrono::high_resolution_clock::now();
        }

        if (count%2 == 1) {
            update(slave_odd,slave_even,down_bound,up_bound);
            maintain_fire(slave_even,slave_fire_area);
            maintain_wall(slave_even);
        } else {
            update(slave_even,slave_odd,down_bound,up_bound);
            maintain_fire(slave_odd,slave_fire_area);
            maintain_wall(slave_odd);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        if (my_rank == 0){
            t2 = std::chrono::high_resolution_clock::now();
            this_time = std::chrono::duration<double>(t2 - t1).count();
            total_time += this_time;
            printf("Iteration %d, elapsed time: %.6f\n", count, this_time);    
        }        
        count++;

        MPI_Gather(slave_odd,(size/world_size)*size,MPI_FLOAT,data_odd,(size/world_size)*size,MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Gather(slave_even,(size/world_size)*size,MPI_FLOAT,data_even,(size/world_size)*size,MPI_FLOAT,0,MPI_COMM_WORLD);
        if (my_rank == 0){
            #ifdef GUI
            if (count % 2 == 1) {
                data2pixels(data_even, pixels);
            } else {
                data2pixels(data_odd, pixels);
            }
            plot(pixels);
            #endif
        }
    }

    delete[] data_odd;
    delete[] data_even;
    delete[] fire_area;
    delete[] slave_odd;
    delete[] slave_even;
    delete[] slave_fire_area;

    delete[] down_bound;
    delete[] up_bound;

    #ifdef GUI
    delete[] pixels;
    #endif
}


int main(int argc, char *argv[]) {
    size = atoi(argv[1]);
    n_omp_threads = atoi(argv[2]);

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (my_rank == 0) {
        #ifdef GUI
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(resolution, resolution);
        glutCreateWindow("Heat Distribution Simulation bonus Implementation");
        gluOrtho2D(0, resolution, 0, resolution);
        #endif    
	}

    master();

	if (my_rank == 0){
		printf("Student ID: 120090414\n"); // replace it with your student id
		printf("Name: Xiang Fei\n"); // replace it with your name
		printf("Assignment 4: Heat Distribution Simulation MPI + OpenMP Implementation\n");
	}

	MPI_Finalize();

	return 0;
}

