#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <gmpxx.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N;

    // Read N only on the root process
    if (rank == 0) {
        std::cout << "Enter the number of steps (N): ";
        std::cin >> N;
    }

    // Broadcast N to all other processes
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // All processes calculate the precision
    long decimals = 2 * std::log10(N) + 5;
    mpf_set_default_prec(decimals * 3.32);

    if (rank == 0) {
        std::cout << "Automatically allocated memory for " << decimals << " decimal places.\n";
    }

    mpf_class mpf_N(std::to_string(N));
    mpf_class step = 1.0;
    step /= mpf_N;

    long long chunk = N / size;
    long long remainder = N % size;

    long long my_chunk = chunk + (rank < remainder ? 1 : 0);
    long long start = rank * chunk + (rank < remainder ? rank : remainder);
    long long end = start + my_chunk;

    MPI_Barrier(MPI_COMM_WORLD);
    // START TIMER
    double start_time = MPI_Wtime();

    mpf_t local_sum_t, x_t, term_t, temp_t, mpf_step, c_4, c_1, c_05;
    mpf_inits(local_sum_t, x_t, term_t, temp_t, mpf_step, c_4, c_1, c_05, NULL);

    mpf_set_ui(local_sum_t, 0);
    mpf_set(mpf_step, step.get_mpf_t());
    mpf_set_d(c_4, 4.0);
    mpf_set_d(c_1, 1.0);
    mpf_set_d(c_05, 0.5);

    for (long long i = start; i < end; i++) {
        mpf_set_d(x_t, (double)i);
        mpf_add(x_t, x_t, c_05);
        mpf_mul(x_t, x_t, mpf_step);

        mpf_mul(temp_t, x_t, x_t);
        mpf_add(temp_t, temp_t, c_1);

        mpf_div(term_t, c_4, temp_t);
        mpf_add(local_sum_t, local_sum_t, term_t);
    }

    mpf_class local_sum(local_sum_t);
    mpf_clears(local_sum_t, x_t, term_t, temp_t, mpf_step, c_4, c_1, c_05, NULL);


    // Serialize GMP number to string for sending
    char* str_ptr = NULL;
    gmp_asprintf(&str_ptr, "%.*Ff", decimals + 5, local_sum.get_mpf_t());
    std::string s(str_ptr);
    free(str_ptr);
    int len = s.length() + 1;

    if (rank == 0) {
        mpf_class total_sum = local_sum;

        for (int i = 1; i < size; i++) {
            int recv_len;
            MPI_Recv(&recv_len, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<char> buf(recv_len);
            MPI_Recv(buf.data(), recv_len, MPI_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Deserialize
            mpf_class recv_sum(buf.data());
            total_sum += recv_sum;
        }

        mpf_class pi = total_sum * step;

        // STOP TIMER
        double end_time = MPI_Wtime();

        std::cout << "\nResult (MPI):\n";
        gmp_printf("Pi: %.*Ff\n", decimals, pi.get_mpf_t());
        std::cout << "Execution time: " << end_time - start_time << " seconds\n";
    }
    else {
        // Send string length, then the string itself
        MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(s.c_str(), len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}