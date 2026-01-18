#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int coresPerSM = 0;
        // Compute number of cores per SM based on architecture
        switch (prop.major) {
            case 2: // Fermi
                coresPerSM = (prop.minor == 1) ? 48 : 32; break;
            case 3: coresPerSM = 192; break; // Kepler
            case 5: coresPerSM = 128; break; // Maxwell
            case 6: coresPerSM = (prop.minor == 1 || prop.minor == 2) ? 128 : 64; break; // Pascal
            case 7: coresPerSM = (prop.minor == 0 || prop.minor == 5) ? 64 : 128; break; // Volta/Turing
            case 8: coresPerSM = (prop.minor == 0) ? 64 : 128; break; // Ampere
            case 9: coresPerSM = 128; break; // Ada / Hopper
            default: coresPerSM = 0; break;
        }

        int totalCores = coresPerSM * prop.multiProcessorCount;

        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  CUDA Cores per SM: " << coresPerSM << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Total CUDA Cores: " << totalCores << std::endl << std::endl;
    }

    return 0;
}