#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Function to convert decibels to power
double db2pow(double db) {
    return std::pow(10.0, db / 10.0);
}

// Function to convert power to decibels
double pow2db(double pow) {
    return 10.0 * std::log10(pow);
}

// C++ implementation of the CFAR algorithm
std::vector<std::vector<double>> cfar_detector(
    const std::vector<std::vector<double>>& rdm,
    int training_cells_range,
    int training_cells_doppler,
    int guard_cells_range,
    int guard_cells_doppler,
    double offset_db) {

    int num_range_bins = rdm.size();
    int num_doppler_bins = rdm[0].size();

    std::vector<std::vector<double>> cfar_signal(num_range_bins, std::vector<double>(num_doppler_bins, 0.0));

    int radius_range = training_cells_range + guard_cells_range;
    int radius_doppler = training_cells_doppler + guard_cells_doppler;

    for (int r = radius_range; r < num_range_bins - radius_range; ++r) {
        for (int d = radius_doppler; d < num_doppler_bins - radius_doppler; ++d) {
            double noise_level = 0.0;
            int cell_count = 0;

            for (int dr = -radius_range; dr <= radius_range; ++dr) {
                for (int dd = -radius_doppler; dd <= radius_doppler; ++dd) {
                    // Check if the cell is a training cell
                    if (std::abs(dr) > guard_cells_range || std::abs(dd) > guard_cells_doppler) {
                        noise_level += db2pow(rdm[r + dr][d + dd]);
                        cell_count++;
                    }
                }
            }

            double average_noise = noise_level / cell_count;
            double threshold = pow2db(average_noise) + offset_db;

            if (rdm[r][d] >= threshold) {
                cfar_signal[r][d] = rdm[r][d];
            }
        }
    }

    return cfar_signal;
}

int main() {
    // Example Usage
    int num_range_bins = 100;
    int num_doppler_bins = 100;

    // Create a sample RDM with a target
    std::vector<std::vector<double>> rdm(num_range_bins, std::vector<double>(num_doppler_bins, 10.0));
    rdm[50][50] = 30.0; // Target

    int training_cells_range = 8;
    int training_cells_doppler = 8;
    int guard_cells_range = 4;
    int guard_cells_doppler = 4;
    double offset_db = 8.0;

    std::vector<std::vector<double>> cfar_output = cfar_detector(
        rdm,
        training_cells_range,
        training_cells_doppler,
        guard_cells_range,
        guard_cells_doppler,
        offset_db
    );

    std::cout << "CFAR Output:" << std::endl;
    for (int r = 0; r < num_range_bins; ++r) {
        for (int d = 0; d < num_doppler_bins; ++d) {
            if (cfar_output[r][d] > 0) {
                std::cout << "Target detected at (" << r << ", " << d << ") with value " << cfar_output[r][d] << std::endl;
            }
        }
    }

    return 0;
}
