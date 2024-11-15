#include <iostream>
#include "DILU_forward_1x1_kernel.h"
#include <chrono>
#include <fstream>
#include <vector>


#include <amgx_types/util.h>
#include <solvers/multicolor_dilu_solver.h>


static int parse_arguments(int argc, char* argv[], int& fileNo, int& test_cycle, int& warmup_cycle)
{
    if (argc >= 2)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if ((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if ((arg == "-h") || (arg == "--help"))
                {
                    return EXIT_FAILURE;
                }
                if ((arg == "-f") && (i + 1 < argc))
                {
                    fileNo = atoi(argv[++i]);
                }
                else if ((arg == "-tc") && (i + 1 < argc))
                {
                    test_cycle = atoi(argv[++i]);
                }
                else if ((arg == "-wc") && (i + 1 < argc))
                {
                    warmup_cycle = atoi(argv[++i]);
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

template<typename T>
std::vector<T> readDataFromBin(const std::string& filename, int byte, int n)
{
    std::vector<T> data;
    std::ifstream  inFile(filename, std::ios::binary);
    if (inFile.is_open())
    {
        inFile.seekg(byte); // locate byte x
        T value;
        for (int i = 0; i < n; ++i)
        {
            inFile.read(reinterpret_cast<char*>(&value), sizeof(T));
            data.push_back(value);
        }
        inFile.close();
    }
    else
    {
        std::cerr << "Cannot open file." << std::endl;
    }
    return data;
}

int main(int argc, char* argv[])
{
   
    int N           = 0;
    int warmup_cycle = 10;
    int test_cycle = 10;
    if (parse_arguments(argc, argv, N, test_cycle, warmup_cycle))
    {
        return EXIT_FAILURE;
    }

    std::string filename = "../data/data_cage14_DILU_forward_1x1_kernel.bin";
    if (N != 0)
    {
        filename = filename + "_" + std::to_string(N);  
    }
    printf("Data %d file is used.\n", N);


    int pos = 0;
    int A_rows_n = readDataFromBin<int>(filename, pos, 1).at(0);
    std::vector<int> A_rows_h = readDataFromBin<int>(filename, pos+=sizeof(int), A_rows_n);
    int* A_rows_d;
    musaMalloc(&A_rows_d, A_rows_n * sizeof(int));
    musaMemcpy(A_rows_d, A_rows_h.data(), A_rows_n * sizeof(int), musaMemcpyHostToDevice);
    int* A_rows_d_tmp;
    musaMalloc(&A_rows_d_tmp, A_rows_n * sizeof(int));

   
    int A_cols_n = readDataFromBin<int>(filename, pos+=sizeof(int)*A_rows_n, 1).at(0);
    std::vector<int> A_cols_h = readDataFromBin<int>(filename, pos+=sizeof(int), A_cols_n);
    int* A_cols_d;
    musaMalloc(&A_cols_d, A_cols_n * sizeof(int));
    musaMemcpy(A_cols_d, A_cols_h.data(), A_cols_n * sizeof(int), musaMemcpyHostToDevice);
    int *A_cols_d_tmp;
    musaMalloc(&A_cols_d_tmp, A_cols_n * sizeof(int));


    int A_vals_n = readDataFromBin<int>(filename, pos+=sizeof(int)*A_cols_n, 1).at(0);
    std::vector<float> A_vals_h = readDataFromBin<float>(filename, pos+=sizeof(int), A_vals_n);
    float* A_vals_d;
    musaMalloc(&A_vals_d, A_vals_n * sizeof(float));
    musaMemcpy(A_vals_d, A_vals_h.data(), A_vals_n * sizeof(float), musaMemcpyHostToDevice);
    float *A_vals_d_tmp;
    musaMalloc(&A_vals_d_tmp, A_vals_n * sizeof(float));


    int A_diag_n = readDataFromBin<int>(filename, pos+=sizeof(float)*A_vals_n, 1).at(0);
    std::vector<int> A_diag_h = readDataFromBin<int>(filename, pos+=sizeof(int), A_diag_n);
    int* A_diag_d;
    musaMalloc(&A_diag_d, A_diag_n * sizeof(int));
    musaMemcpy(A_diag_d, A_diag_h.data(), A_diag_n * sizeof(int), musaMemcpyHostToDevice);
    int* A_diag_d_tmp;
    musaMalloc(&A_diag_d_tmp, A_diag_n * sizeof(int));

    int x_n = readDataFromBin<int>(filename, pos+=sizeof(int)*A_diag_n, 1).at(0);
    std::vector<float> x_h= readDataFromBin<float>(filename, pos+=sizeof(int), x_n);
    float* x_d;
    musaMalloc(&x_d, x_n * sizeof(float));
    musaMemcpy(x_d, x_h.data(), x_n * sizeof(float), musaMemcpyHostToDevice);
    float* x_d_tmp;
    musaMalloc(&x_d_tmp, x_n * sizeof(float));


    int b_n = readDataFromBin<int>(filename, pos+=sizeof(float)*x_n, 1).at(0);
    std::vector<float> b_h = readDataFromBin<float>(filename, pos+=sizeof(int), b_n);
    float* b_d;
    musaMalloc(&b_d, b_n * sizeof(float));
    musaMemcpy(b_d, b_h.data(), b_n * sizeof(float), musaMemcpyHostToDevice);
    float* b_d_tmp;
    musaMalloc(&b_d_tmp, b_n * sizeof(float));


    int delta_n = readDataFromBin<int>(filename, pos+=sizeof(float)*b_n, 1).at(0);
    std::vector<float> delta_h = readDataFromBin<float>(filename, pos+=sizeof(int), delta_n);
    float* delta_d;
    musaMalloc(&delta_d, delta_n * sizeof(float));
    musaMemcpy(delta_d, delta_h.data(), delta_n * sizeof(float), musaMemcpyHostToDevice);
    float* delta_d_tmp;
    musaMalloc(&delta_d_tmp, delta_n * sizeof(float));


    int sorted_rows_by_color_n = readDataFromBin<int>(filename, pos+=sizeof(float)*delta_n, 1).at(0);
    std::vector<int> sorted_rows_by_color_h = readDataFromBin<int>(filename, pos+=sizeof(int), sorted_rows_by_color_n);
    int* sorted_rows_by_color_d;
    musaMalloc(&sorted_rows_by_color_d, sorted_rows_by_color_n * sizeof(int));
    musaMemcpy(sorted_rows_by_color_d, sorted_rows_by_color_h.data(), sorted_rows_by_color_n * sizeof(int), musaMemcpyHostToDevice);
    int* sorted_rows_by_color_d_tmp;
    musaMalloc(&sorted_rows_by_color_d_tmp, sorted_rows_by_color_n * sizeof(int));


    int num_rows_per_color_h = readDataFromBin<int>(filename, pos+=sizeof(int)*sorted_rows_by_color_n, 1).at(0);

    int current_color_h = readDataFromBin<int>(filename, pos+=sizeof(int), 1).at(0);

    int row_colors_n = readDataFromBin<int>(filename, pos+=sizeof(int), 1).at(0);
    std::vector<int> row_colors_h = readDataFromBin<int>(filename, pos+=sizeof(int), row_colors_n);
    int* row_colors_d;
    musaMalloc(&row_colors_d, row_colors_n * sizeof(int));
    musaMemcpy(row_colors_d, row_colors_h.data(), row_colors_n * sizeof(int), musaMemcpyHostToDevice);
    int* row_colors_d_tmp;
    musaMalloc(&row_colors_d_tmp, row_colors_n * sizeof(int));


    int Einv_n = readDataFromBin<int>(filename, pos+=sizeof(int)*row_colors_n, 1).at(0);
    std::vector<float> Einv_h = readDataFromBin<float>(filename, pos+=sizeof(int), Einv_n);
    float* Einv_d;
    musaMalloc(&Einv_d, Einv_n * sizeof(float));
    musaMemcpy(Einv_d, Einv_h.data(), Einv_n * sizeof(float), musaMemcpyHostToDevice);
    float* Einv_d_tmp;
    musaMalloc(&Einv_d_tmp, Einv_n * sizeof(float));


    amgx::ColoringType boundary_coloring_h = readDataFromBin<amgx::ColoringType>(
                                             filename, pos+=sizeof(float)*Einv_n, 1).at(0);

    int boundary_index_h = readDataFromBin<int>(filename, pos+=sizeof(amgx::ColoringType), 1).at(0);


    const int NUM_THREADS_PER_ROW = 8;//readDataFromBin<int>(filename, pos+=sizeof(int), 1).at(0);
    const int CTA_SIZE = 128;//readDataFromBin<int>(filename, pos+=sizeof(int), 1).at(0);
    const int WARP_SIZE = 32;//readDataFromBin<int>(filename, pos+=sizeof(int), 1).at(0);
    int grid_size = readDataFromBin<int>(filename, pos+=sizeof(int)*4, 1).at(0);

    // const int NUM_THREADS_PER_ROW = 8;
    // const int NUM_ROWS_PER_CTA = CTA_SIZE / NUM_THREADS_PER_ROW;
    // const int grid_size = std::min( 4096, (num_rows_per_color + NUM_ROWS_PER_CTA - 1) / NUM_ROWS_PER_CTA );


    double time_sum  = 0.0;
    int total_cycle = warmup_cycle + test_cycle;
    for(int i = 0; i < total_cycle; ++i)
    {
        musaMemcpy(A_rows_d_tmp, A_rows_d, A_rows_n * sizeof(int), musaMemcpyDeviceToDevice);
        musaMemcpy(A_cols_d_tmp, A_cols_d, A_cols_n * sizeof(int), musaMemcpyDeviceToDevice);
        musaMemcpy(A_vals_d_tmp, A_vals_d, A_vals_n * sizeof(float), musaMemcpyDeviceToDevice);
        musaMemcpy(A_diag_d_tmp, A_diag_d, A_diag_n * sizeof(int), musaMemcpyDeviceToDevice);
        musaMemcpy(x_d_tmp, x_d, x_n * sizeof(float), musaMemcpyDeviceToDevice);
        musaMemcpy(b_d_tmp, b_d, b_n * sizeof(float), musaMemcpyDeviceToDevice);
        musaMemcpy(delta_d_tmp, delta_d, delta_n * sizeof(float), musaMemcpyDeviceToDevice);
        musaMemcpy(sorted_rows_by_color_d_tmp, sorted_rows_by_color_d, sorted_rows_by_color_n * sizeof(int), musaMemcpyDeviceToDevice);
        musaMemcpy(row_colors_d_tmp, row_colors_d, row_colors_n * sizeof(int), musaMemcpyDeviceToDevice);
        musaMemcpy(Einv_d_tmp, Einv_d, Einv_n * sizeof(float), musaMemcpyDeviceToDevice);

        if (i < warmup_cycle) {
            DILU_forward_1x1_kernel<float, float, NUM_THREADS_PER_ROW, CTA_SIZE, WARP_SIZE, false> <<< grid_size, CTA_SIZE>>>(
                        A_rows_d_tmp, A_cols_d_tmp, A_vals_d_tmp, A_diag_d_tmp, x_d_tmp, b_d_tmp, delta_d_tmp, sorted_rows_by_color_d_tmp, 
                        num_rows_per_color_h, current_color_h, row_colors_d_tmp, Einv_d_tmp, boundary_coloring_h, boundary_index_h);
        } else {
            auto starttime_single = std::chrono::system_clock::now();
            DILU_forward_1x1_kernel<float, float, NUM_THREADS_PER_ROW, CTA_SIZE, WARP_SIZE, false> <<< grid_size, CTA_SIZE>>>(
                        A_rows_d_tmp, A_cols_d_tmp, A_vals_d_tmp, A_diag_d_tmp, x_d_tmp, b_d_tmp, delta_d_tmp, sorted_rows_by_color_d_tmp, 
                        num_rows_per_color_h, current_color_h, row_colors_d_tmp, Einv_d_tmp, boundary_coloring_h, boundary_index_h);
            musaStreamSynchronize(0);
            auto endtime_single = std::chrono::system_clock::now();
            auto single_count = std::chrono::duration_cast<std::chrono::microseconds>(endtime_single - starttime_single).count();
            time_sum += double(single_count * std::chrono::microseconds::period::num);
        }
    }

    auto time_average = time_sum / test_cycle;
    printf("file %d grid_size:  %d\n", N, grid_size);
    printf("warmup %d times, average of %d times\n", warmup_cycle, test_cycle);
    printf("DILU_forward_1x1_kernel Time Used:  %f us\n", time_average);

    musaFree(A_rows_d);
    musaFree(A_cols_d);
    musaFree(A_vals_d);
    musaFree(A_diag_d);
    musaFree(x_d);
    musaFree(b_d);
    musaFree(delta_d);
    musaFree(sorted_rows_by_color_d);
    musaFree(row_colors_d);
    musaFree(Einv_d);

    musaFree(A_rows_d_tmp);
    musaFree(A_cols_d_tmp);
    musaFree(A_vals_d_tmp);
    musaFree(A_diag_d_tmp);
    musaFree(x_d_tmp);
    musaFree(b_d_tmp);
    musaFree(delta_d_tmp);
    musaFree(sorted_rows_by_color_d_tmp);
    musaFree(row_colors_d_tmp);
    musaFree(Einv_d_tmp);

    return EXIT_SUCCESS;
}
