#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"


int n_body;
int n_iteration;


int my_rank;
int world_size;

std::chrono::high_resolution_clock::time_point t1;
std::chrono::high_resolution_clock::time_point t2;
std::chrono::duration<double> time_span;
Logger l = Logger("mpi", n_body, bound_x, bound_y);


void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n) {
    // TODO: Generate proper initial position and mass for better visualization
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % max_mass + 1.0f;
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);
        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}


void update_position(double *x, double *y, double *vx, double *vy, int n) {
    //TODO: update position 
    for(int i=0;i<n;i++){
        x[i] = x[i] + vx[i]*dt;
        y[i] = y[i] + vy[i]*dt;
        // check the ball & wall collision
        if(x[i]<0){
            vx[i] = -vx[i];
            x[i] = -x[i];
        }
        else if(x[i]>bound_x){
            vx[i] = -vx[i];
            x[i] = 2*bound_x-x[i];
        }
        if(y[i]<0){
            vy[i] = -vy[i];
            y[i] = -y[i];
        }
        else if(y[i]>bound_y){
            vy[i] = -vy[i];
            y[i] = 2*bound_y-y[i];
        }
    }
}


void update_velocity(double *local_vx, double *local_vy, int local_n, double *m, double *x, double *y, double *vx, double *vy, int n) {
    //TODO: calculate force and acceleration, update velocity
    for(int i=0;i<local_n;i++){
        double axi = 0;
        double ayi = 0;
        for(int j=0;j<n;j++){
            if(j==i+my_rank*local_n){
                continue;
            }
            double distance2 = pow(x[i]-x[j],2)+pow(y[i]-y[j],2);
            if(distance2<=4*radius2){ // ball i and j have a collision
                if(local_vx[i]*local_vx[j]<0 || (local_vx[i]>=0&&local_vx[j]>=0&&local_vx[i]>local_vx[j]) || (local_vx[i]<=0&&local_vx[j]<=0&&local_vx[i]>local_vx[j])){
                    local_vx[i] = ((m[i]-m[j])*local_vx[i]+2*m[j]*local_vx[j])/(m[i]+m[j]);
                    local_vx[j] = ((m[j]-m[i])*local_vx[i]+2*m[i]*local_vx[i])/(m[i]+m[j]);
                }
                if(vy[i]*vy[j]<0 || (vy[i]>=0&&vy[j]>=0&&vy[i]>vy[j]) || (vy[i]<=0&&vy[j]<=0&&vy[i]>vy[j])){
                    local_vy[i] = ((m[i]-m[j])*local_vy[i]+2*m[j]*local_vy[j])/(m[i]+m[j]);
                    local_vy[j] = ((m[j]-m[i])*local_vy[i]+2*m[i]*local_vy[i])/(m[i]+m[j]);
                }
                axi = 0;
                ayi = 0;
                break;
            }
            else{
                double forcex = gravity_const*m[i]*m[j]*(x[j]-x[i])/(pow(distance2,1.5)+err);
                double forcey = gravity_const*m[i]*m[j]*(y[j]-y[i])/(pow(distance2,1.5)+err);
                axi += forcex / m[i];
                ayi += forcey / m[i];
            }
        }
        local_vx[i] = local_vx[i] + axi*dt;
        local_vy[i] = local_vy[i] + ayi*dt;
    }
}


void mpi_run() {
    double* total_m = new double[n_body];
    double* total_x = new double[n_body];
    double* total_y = new double[n_body];
    double* total_vx = new double[n_body];
    double* total_vy = new double[n_body];
    int my_length = n_body / world_size;
    int remainder = n_body % world_size;

    double* local_m = new double[my_length];
    double* local_x = new double[my_length];
    double* local_y = new double[my_length];
    double* local_vx = new double[my_length];
    double* local_vy = new double[my_length];

    if(my_rank==0){
        generate_data(total_m, total_x, total_y, total_vx, total_vy, n_body);   
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(total_m, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(total_m, my_length, MPI_DOUBLE, local_m, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n_iteration; i++){
        if(my_rank==0){
            t1 = std::chrono::high_resolution_clock::now();
        }
        // TODO: MPI routine
        
        MPI_Bcast(total_x, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(total_y, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(total_vx, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(total_vy, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(total_x, my_length, MPI_DOUBLE, local_x, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(total_y, my_length, MPI_DOUBLE, local_y, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(total_vx, my_length, MPI_DOUBLE, local_vx, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(total_vy, my_length, MPI_DOUBLE, local_vy, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        update_velocity(local_vx,local_vy,my_length,total_m,total_x,total_y,total_vx,total_vy,n_body);
        if(my_rank==0){
            for(int i=n_body-remainder;i<n_body;i++){
                double axi = 0;
                double ayi = 0;
                for(int j=0;j<n_body;j++){
                    if(j==i){
                        continue;
                    }
                    double distance2 = pow(total_x[i]-total_x[j],2)+pow(total_y[i]-total_y[j],2);
                    if(distance2<=4*radius2){ // ball i and j have a collision
                        if(total_vx[i]*total_vx[j]<0 || (total_vx[i]>=0&&total_vx[j]>=0&&total_vx[i]>total_vx[j]) || (total_vx[i]<=0&&total_vx[j]<=0&&total_vx[i]>total_vx[j])){
                            total_vx[i] = ((total_m[i]-total_m[j])*total_vx[i]+2*total_m[j]*total_vx[j])/(total_m[i]+total_m[j]);
                            total_vx[j] = ((total_m[j]-total_m[i])*total_vx[i]+2*total_m[i]*total_vx[i])/(total_m[i]+total_m[j]);
                        }
                        if(total_vy[i]*total_vy[j]<0 || (total_vy[i]>=0&&total_vy[j]>=0&&total_vy[i]>total_vy[j]) || (total_vy[i]<=0&&total_vy[j]<=0&&total_vy[i]>total_vy[j])){
                            total_vy[i] = ((total_m[i]-total_m[j])*total_vy[i]+2*total_m[j]*total_vy[j])/(total_m[i]+total_m[j]);
                            total_vy[j] = ((total_m[j]-total_m[i])*total_vy[i]+2*total_m[i]*total_vy[i])/(total_m[i]+total_m[j]);
                        }
                        axi = 0;
                        ayi = 0;
                        break;
                    }
                    else{
                        double forcex = gravity_const*total_m[i]*total_m[j]*(total_x[j]-total_x[i])/(pow(distance2,1.5)+err);
                        double forcey = gravity_const*total_m[i]*total_m[j]*(total_y[j]-total_y[i])/(pow(distance2,1.5)+err);
                        axi += forcex / total_m[i];
                        ayi += forcey / total_m[i];
                    }
                }
                total_vx[i] = total_vx[i] + axi*dt;
                total_vy[i] = total_vy[i] + ayi*dt;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        update_position(local_x,local_y,local_vx,local_vy,my_length);
        if(my_rank==0){
            for(int i=n_body-remainder;i<n_body;i++){
                total_x[i] = total_x[i] + total_vx[i]*dt;
                total_y[i] = total_y[i] + total_vy[i]*dt;
                // check the ball & wall collision
                if(total_x[i]<0){
                    total_vx[i] = -total_vx[i];
                    total_x[i] = -total_x[i];
                }
                else if(total_x[i]>bound_x){
                    total_vx[i] = -total_vx[i];
                    total_x[i] = 2*bound_x-total_x[i];
                }
                if(total_y[i]<0){
                    total_vy[i] = -total_vy[i];
                    total_y[i] = -total_y[i];
                }
                else if(total_y[i]>bound_y){
                    total_vy[i] = -total_vy[i];
                    total_y[i] = 2*bound_y-total_y[i];
                }
            }
        }

        MPI_Gather(local_x, my_length, MPI_DOUBLE, total_x, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_y, my_length, MPI_DOUBLE, total_y, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_vx, my_length, MPI_DOUBLE, total_vx, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_vy, my_length, MPI_DOUBLE, total_vy, my_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // TODO End
        
        if(my_rank==0){
            t2 = std::chrono::high_resolution_clock::now();
            time_span = t2 - t1;

            printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

            l.save_frame(total_x, total_y);

            #ifdef GUI
            glClear(GL_COLOR_BUFFER_BIT);
            glColor3f(1.0f, 0.0f, 0.0f);
            glPointSize(2.0f);
            glBegin(GL_POINTS);
            double xi;
            double yi;
            for (int i = 0; i < n_body; i++){
                xi = total_x[i];
                yi = total_y[i];
                glVertex2f(xi, yi);
            }
            glEnd();
            glFlush();
            glutSwapBuffers();
            #else

            #endif
        }
    }

    delete total_m;
    delete total_x;
    delete total_y;
    delete total_vx;
    delete total_vy;

    delete local_m;
    delete local_x;
    delete local_y;
    delete local_vx;
    delete local_vy;

}




int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (my_rank == 0) {
		#ifdef GUI
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(500, 500); 
		glutInitWindowPosition(0, 0);
		glutCreateWindow("N Body Simulation MPI Implementation");
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0, bound_x, 0, bound_y);
		#endif
	}

    mpi_run();

	if (my_rank == 0){
		printf("Student ID: 120090414\n"); // replace it with your student id
		printf("Name: Xiang Fei\n"); // replace it with your name
		printf("Assignment 2: N Body Simulation MPI Implementation\n");
	}

	MPI_Finalize();

	return 0;
}

