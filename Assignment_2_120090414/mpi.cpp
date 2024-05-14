#include "asg2.h"
#include <stdio.h>
#include <mpi.h>


int rank;
int world_size;


void mpi_compute() {
	//TODO: procedure to compute
	// Create MPI variable types to send information with non-uniform data types at once
	MPI_Aint displacement[3] = {0,4,8};
	int blockLength[3] = {1,1,1};
	int types[3] = { MPI_INT, MPI_INT, MPI_FLOAT };
	MPI_Datatype mpi_pointtype;
	MPI_Type_create_struct(
		3,
		blockLength,
		displacement,
		types,
		&mpi_pointtype
	);
	MPI_Type_commit(&mpi_pointtype);

	int my_length = X_RESN / world_size;
	int remainder = X_RESN % world_size;

	Point my_point[my_length*Y_RESN];
	MPI_Scatter(data,my_length*Y_RESN,mpi_pointtype,my_point,my_length*Y_RESN,mpi_pointtype,0,MPI_COMM_WORLD); 

	for (int i=0; i<my_length*Y_RESN; i++) {
		compute(&my_point[i]);
	}

	if(rank==0){
		for (int i=0;i<remainder;i++){
			compute(&data[my_length*Y_RESN*world_size+i]);
		}
	}

	MPI_Gather(my_point,my_length*Y_RESN,mpi_pointtype,data,my_length*Y_RESN,mpi_pointtype,0,MPI_COMM_WORLD);
	
	MPI_Type_free(&mpi_pointtype);
	//TODO END
}



int main(int argc, char *argv[]) {
	if ( argc == 4 ) {
		X_RESN = atoi(argv[1]);
		Y_RESN = atoi(argv[2]);
		max_iteration = atoi(argv[3]);
	} else {
		X_RESN = 1000;
		Y_RESN = 1000;
		max_iteration = 100;
	}

	if (rank == 0) {
		#ifdef GUI
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(500, 500); 
		glutInitWindowPosition(0, 0);
		glutCreateWindow("MPI");
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0, X_RESN, 0, Y_RESN);
		glutDisplayFunc(plot);
		#endif
	}

	/* computation part begin */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	

	if (rank == 0){
		initData();
		t1 = std::chrono::high_resolution_clock::now();
	}

	
	mpi_compute();

	
	if (rank == 0){
		t2 = std::chrono::high_resolution_clock::now();  
		time_span = t2 - t1;
	}

	if (rank == 0){
		printf("Student ID: 120090414\n"); // replace it with your student id
		printf("Name: Xiang Fei\n"); // replace it with your name
		printf("Assignment 2 MPI\n");
		printf("Run Time: %f seconds\n", time_span.count());
		printf("Problem Size: %d * %d, %d\n", X_RESN, Y_RESN, max_iteration);
		printf("Process Number: %d\n", world_size);
	}

	MPI_Finalize();
	/* computation part end */

	if (rank == 0){
		#ifdef GUI
		glutMainLoop();
		#endif
	}

	return 0;
}

