#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>
#include <limits.h>


int main (int argc, char **argv){

    MPI_Init(&argc, &argv); 

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_elements; // number of elements to be sorted
    
    num_elements = atoi(argv[1]); // convert command line argument to num_elements
    int num_my_element = num_elements / world_size; // number of elements allocated to each process
    int remainder = num_elements - num_my_element*world_size;
    int num_elements_extended = num_elements+world_size-remainder;

    int elements_extended[num_elements_extended];
    int sorted_elements[num_elements]; // store sorted elements
    int sorted_elements_extended[num_elements_extended];
    

    if (rank == 0) { // read inputs from file (master process)
        std::ifstream input(argv[2]);
        int element;
        int i = 0;
        while (input >> element) {
            elements_extended[i] = element;
            i++;
        }
        for(int i=num_elements;i<num_elements_extended;i++){
            elements_extended[i] = INT_MAX;
        }
        std::cout << "actual number of elements:" << i << std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    if (rank == 0){ 
        t1 = std::chrono::high_resolution_clock::now(); // record time
    }

    /* TODO BEGIN
        Implement parallel odd even transposition sort
        Code in this block is not a necessary. 
        Replace it with your own code.
        Useful MPI documentation: https://rookiehpc.github.io/mpi/docs
    */

    num_my_element = num_elements_extended/world_size;
    int my_element[num_my_element]; // store elements of each process
    MPI_Scatter(elements_extended, num_my_element, MPI_INT, my_element, num_my_element, MPI_INT, 0, MPI_COMM_WORLD); // distribute elements to each process
    
    int send_element_head;
    int send_element_tail;
    int recv_element;

    for(int i=0;i<num_elements_extended;i++){
        if(i%2==0){
            for(int j=0;j<num_my_element-1;j+=2){
                if(my_element[j]>my_element[j+1]){
                    std::swap(my_element[j],my_element[j+1]);
                }
            }
        }
        else{
            for(int j=1;j<num_my_element-1;j+=2){
                if(my_element[j]>my_element[j+1]){
                    std::swap(my_element[j],my_element[j+1]);
                }
            }
            if(rank!=0){
                send_element_head = my_element[0];
                MPI_Sendrecv(&send_element_head,1,MPI_LONG,rank-1,0,&recv_element,1,MPI_LONG,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                if(recv_element>my_element[0]){
                    my_element[0]=recv_element;
                }
            }
            if(rank!=world_size-1){
                send_element_tail = my_element[num_my_element-1];
                MPI_Sendrecv(&send_element_tail,1,MPI_LONG,rank+1,0,&recv_element,1,MPI_LONG,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                if(my_element[num_my_element-1]>recv_element){
                    my_element[num_my_element-1]=recv_element;
                }
            }
        }
    }

    MPI_Gather(my_element, num_my_element, MPI_INT, sorted_elements_extended, num_my_element, MPI_INT, 0, MPI_COMM_WORLD); // collect result from each process
    

    /* TODO END */

    if (rank == 0){ // record time (only executed in master process)
        t2 = std::chrono::high_resolution_clock::now();  
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Student ID: " << "120090414" << std::endl; // replace it with your student id
        std::cout << "Name: " << "Xiang Fei" << std::endl; // replace it with your name
        std::cout << "Assignment 1" << std::endl;
        std::cout << "Run Time: " << time_span.count() << " seconds" << std::endl;
        std::cout << "Input Size: " << num_elements << std::endl;
        std::cout << "Process Number: " << world_size << std::endl; 
    }

    for(int i=0;i<num_elements;i++){
        sorted_elements[i] = sorted_elements_extended[i];
    }

    if (rank == 0){ // write result to file (only executed in master process)
        std::ofstream output(argv[2]+std::string(".parallel.out"), std::ios_base::out);
        for (int i = 0; i < num_elements; i++) {
            output << sorted_elements[i] << std::endl;
        }
    }
    
    MPI_Finalize();
    
    return 0;
}