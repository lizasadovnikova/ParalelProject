#include <iostream>
#include <gmpxx.h>
#include <chrono>
#include <cmath>
#include <string>

int main() {
    long long N;

    std::cout << "Enter the number of steps (N): ";
    std::cin >> N;

    long decimals = 2 * std::log10(N) + 5;
    std::cout << "Automatically allocated memory for " << decimals << " decimal places.\n";

    mpf_set_default_prec(decimals * 3.32);

    mpf_class mpf_N(std::to_string(N));
    mpf_class step = 1.0;
    step /= mpf_N;

    mpf_class total_sum = 0.0;

    // START TIMER
    auto start_time = std::chrono::high_resolution_clock::now();

    mpf_t local_sum, x, term, temp, mpf_step, c_4, c_1, c_05;
    mpf_inits(local_sum, x, term, temp, mpf_step, c_4, c_1, c_05, NULL);

    mpf_set_ui(local_sum, 0);
    mpf_set(mpf_step, step.get_mpf_t());
    mpf_set_d(c_4, 4.0);
    mpf_set_d(c_1, 1.0);
    mpf_set_d(c_05, 0.5);

    for (long long i = 0; i < N; i++) {
        mpf_set_d(x, (double)i);
        mpf_add(x, x, c_05);
        mpf_mul(x, x, mpf_step);

        mpf_mul(temp, x, x);
        mpf_add(temp, temp, c_1);

        mpf_div(term, c_4, temp);

        mpf_add(local_sum, local_sum, term);
    }

    mpf_class temp_class(local_sum);
    total_sum += temp_class;

    mpf_clears(local_sum, x, term, temp, mpf_step, c_4, c_1, c_05, NULL);

    mpf_class pi = total_sum * step;

    // STOP TIMER
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "\nResult (Sequential):\n";
    gmp_printf("Pi: %.*Ff\n", decimals, pi.get_mpf_t());
    std::cout << "Execution time: " << diff.count() << " seconds\n";

    return 0;
}