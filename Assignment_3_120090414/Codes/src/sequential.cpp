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
#include "./headers/logger.h"

int n_body;
int n_iteration;


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

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n) {
    //TODO: calculate force and acceleration, update velocity
    for(int i=0;i<n;i++){
        double axi = 0;
        double ayi = 0;
        for(int j=0;j<n;j++){
            if(j==i){
                continue;
            }
            double distance2 = pow(x[i]-x[j],2)+pow(y[i]-y[j],2);
            if(distance2<=4*radius2){ // ball i and j have a collision
                if(vx[i]*vx[j]<0 || (vx[i]>=0&&vx[j]>=0&&vx[i]>vx[j]) || (vx[i]<=0&&vx[j]<=0&&vx[i]>vx[j])){
                    vx[i] = ((m[i]-m[j])*vx[i]+2*m[j]*vx[j])/(m[i]+m[j]);
                    vx[j] = ((m[j]-m[i])*vx[i]+2*m[i]*vx[i])/(m[i]+m[j]);
                }
                if(vy[i]*vy[j]<0 || (vy[i]>=0&&vy[j]>=0&&vy[i]>vy[j]) || (vy[i]<=0&&vy[j]<=0&&vy[i]>vy[j])){
                    vy[i] = ((m[i]-m[j])*vy[i]+2*m[j]*vy[j])/(m[i]+m[j]);
                    vy[j] = ((m[j]-m[i])*vy[i]+2*m[i]*vy[i])/(m[i]+m[j]);
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
        vx[i] = vx[i] + axi*dt;
        vy[i] = vy[i] + ayi*dt;
    }
}


void master() {
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    Logger l = Logger("sequential", n_body, bound_x, bound_y);

    for (int i = 0; i < n_iteration; i++){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        update_velocity(m, x, y, vx, vy, n_body);
        update_position(x, y, vx, vy, n_body);


        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = t2 - t1;

        printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

        l.save_frame(x, y);

        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        double xi;
        double yi;
        for (int i = 0; i < n_body; i++){
            xi = x[i];
            yi = y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();
        #else

        #endif
    }

    delete m;
    delete x;
    delete y;
    delete vx;
    delete vy;
    
}


int main(int argc, char *argv[]){
    
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation Sequential Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    master();

    printf("Student ID: 120090414\n"); // replace it with your student id
    printf("Name: Xiang Fei\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation Sequential Implementation\n");
    
    return 0;

}
