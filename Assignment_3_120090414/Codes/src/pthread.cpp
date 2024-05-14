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
#include "./headers/logger.h"

int n_thd; // number of threads

pthread_mutex_t mutex;
pthread_barrier_t mybarrier;
int n_body;
int n_iteration;
int count1=0;
int count2=0;

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


void update_position(double *x, double *y, double *vx, double *vy, int n, int tid, int num_thd) {
    //TODO: update position 
    count1 = num_thd-1;
    int current = tid;
    while(count1<n && current<n){
        x[current] = x[current] + vx[current]*dt;
        y[current] = y[current] + vy[current]*dt;
        // check the ball & wall collision
        if(x[current]<0){
            vx[current] = -vx[current];
            x[current] = -x[current];
        }
        else if(x[current]>bound_x){
            vx[current] = -vx[current];
            x[current] = 2*bound_x-x[current];
        }
        if(y[current]<0){
            vy[current] = -vy[current];
            y[current] = -y[current];
        }
        else if(y[current]>bound_y){
            vy[current] = -vy[current];
            y[current] = 2*bound_y-y[current];
        }
        pthread_mutex_lock(&mutex);
        current = count1 + 1;
        count1++;
        pthread_mutex_unlock(&mutex);
        if(count1>=n) break;
    }
}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n, int tid, int num_thd) {
    //TODO: calculate force and acceleration, update velocity
    count2 = num_thd-1;
    int current = tid;
    while(count2<n && current<n){
        double axi = 0;
        double ayi = 0;
        for(int j=0;j<n;j++){
            if(j==current){
                continue;
            }
            double distance2 = pow(x[current]-x[j],2)+pow(y[current]-y[j],2);
            if(distance2<=4*radius2){ // ball i and j have a collision
                if(vx[current]*vx[j]<0 || (vx[current]>=0&&vx[j]>=0&&vx[current]>vx[j]) || (vx[current]<=0&&vx[j]<=0&&vx[current]>vx[j])){
                    vx[current] = ((m[current]-m[j])*vx[current]+2*m[j]*vx[j])/(m[current]+m[j]);
                    vx[j] = ((m[j]-m[current])*vx[current]+2*m[current]*vx[current])/(m[current]+m[j]);
                }
                if(vy[current]*vy[j]<0 || (vy[current]>=0&&vy[j]>=0&&vy[current]>vy[j]) || (vy[current]<=0&&vy[j]<=0&&vy[current]>vy[j])){
                    vy[current] = ((m[current]-m[j])*vy[current]+2*m[j]*vy[j])/(m[current]+m[j]);
                    vy[j] = ((m[j]-m[current])*vy[current]+2*m[current]*vy[current])/(m[current]+m[j]);
                }
                axi = 0;
                ayi = 0;
                break;
            }
            else{
                double forcex = gravity_const*m[current]*m[j]*(x[j]-x[current])/(pow(distance2,1.5)+err);
                double forcey = gravity_const*m[current]*m[j]*(y[j]-y[current])/(pow(distance2,1.5)+err);
                axi += forcex / m[current];
                ayi += forcey / m[current];
            }
        }
        vx[current] = vx[current] + axi*dt;
        vy[current] = vy[current] + ayi*dt;
        pthread_mutex_lock(&mutex);
        current = count2 + 1;
        count2++;
        pthread_mutex_unlock(&mutex);
        if(count2>=n) break;
    }
}


typedef struct {
    //TODO: specify your arguments for threads
    double *m;
    double *x;
    double *y;
    double *vx;
    double *vy;
    int n;
    int tid;
    int num_thd;
    //TODO END
} Args;


void* worker(void* args) {
    //TODO: procedure in each threads
    Args* my_arg = (Args*) args;
    double *m = my_arg->m;
    double *x = my_arg->x;
    double *y = my_arg->y;
    double *vx = my_arg->vx;
    double *vy = my_arg->vy;
    int n = my_arg->n;
    int tid = my_arg->tid;
    int num_thd = my_arg->num_thd;
    
    update_velocity(m, x, y, vx, vy, n, tid, num_thd);
    
    pthread_barrier_wait(&mybarrier);

    update_position(x, y, vx, vy, n, tid, num_thd);
    // TODO END
}


void master(){
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    Logger l = Logger("pthread", n_body, bound_x, bound_y);

    for (int i = 0; i < n_iteration; i++){
        pthread_t thds[n_thd]; // thread pool
        Args args[n_thd]; // arguments for all threads
        for (int thd = 0; thd < n_thd; thd++){
            args[thd].tid = thd;
            args[thd].num_thd = n_thd;
            args[thd].m = m;
            args[thd].x = x;
            args[thd].y = y;
            args[thd].vx = vx;
            args[thd].vy = vy;
            args[thd].n = n_body;
        }
        pthread_mutex_init(&mutex,NULL);
        pthread_barrier_init(&mybarrier, NULL, n_thd);
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        //TODO: assign jobs
        for (int thd = 0; thd < n_thd; thd++) pthread_create(&thds[thd], NULL, worker, &args[thd]);
        for (int thd = 0; thd < n_thd; thd++) pthread_join(thds[thd], NULL);

        //TODO End

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

        pthread_mutex_destroy(&mutex);
        pthread_barrier_destroy(&mybarrier);
    }

    delete m;
    delete x;
    delete y;
    delete vx;
    delete vy;
}

int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_thd = atoi(argv[3]);

    #ifdef GUI
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Pthread");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    master();

	return 0;
}
