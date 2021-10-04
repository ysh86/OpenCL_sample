#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>

const size_t ch = 4;
const size_t width = 1024;
const size_t height = 1024 * 2;

const cl::NDRange kernelRangeGlobal(width * height, 1);
const cl::NDRange kernelRangeLocal(256,1);

#define kernelFile "src/hello.cl"
#define kernelName "hello"

int
main(void)
{
    cl_int err = CL_SUCCESS;
    try {
        // Platform
        // --------------------------------------------
        std::cout << "Platform" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::vector<cl::Platform> platforms;
        err |= cl::Platform::get(&platforms);
        size_t n = platforms.size();
        for (size_t i = 0; i < n; i++) {
            std::cout << i << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            static const int names[] = {
                CL_PLATFORM_PROFILE,
                CL_PLATFORM_VERSION,
                CL_PLATFORM_NAME,
                CL_PLATFORM_VENDOR,
                CL_PLATFORM_EXTENSIONS,
            };
            int n_names = sizeof(names) / sizeof(names[0]);
            for (int j = 0; j < n_names; j++) {
                std::string param;
                err |= platforms[i].getInfo(names[j], &param);
                std::cout << param << std::endl;
            }
        }
        if (n == 0) {
            std::cout << "Platform size 0\n";
            return -1;
        }
        size_t i = 0;
        cl::Platform &plat = platforms[i];


        // Device & Context
        // --------------------------------------------
        std::cout << "Device" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::vector<cl::Device> devices;
        err |= plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        size_t n_dev = devices.size();
        for (size_t i = 0; i < n_dev; i++) {
            std::cout << i << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            static const int names[] = {
                CL_DEVICE_NAME,
                CL_DEVICE_VENDOR,
                CL_DEVICE_PROFILE,
                CL_DEVICE_VERSION,
                CL_DRIVER_VERSION,
                CL_DEVICE_OPENCL_C_VERSION,
                CL_DEVICE_EXTENSIONS,
            };
            int n_names = sizeof(names) / sizeof(names[0]);
            for (int j = 0; j < n_names; j++) {
                std::string param;
                err |= devices[i].getInfo(names[j], &param);
                std::cout << param << std::endl;
            }
        }

        i = 0;
        cl::Device &device = devices[i];
        // Context
        cl::Context context(device);

        // Build
        // --------------------------------------------
        std::ifstream from(kernelFile);
        std::string kernelStr((std::istreambuf_iterator<char>(from)),
                               std::istreambuf_iterator<char>());
        from.close();
        cl::Program::Sources sources{kernelStr};
        cl::Program program = cl::Program(context, sources);
        try {
            err |= program.build("");
        } catch (cl::Error &err) {
            std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
            << ")"
            << std::endl;

            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &buildErr);
            std::cerr << buildInfo << std::endl;
            return EXIT_FAILURE;
        }
        cl::Kernel kernel(program, kernelName, &err);

        // Execution
        // --------------------------------------------
        cl::Buffer a(context, CL_MEM_READ_ONLY, width*ch*height);
        cl::Buffer b(context, CL_MEM_READ_ONLY, width*ch*height);
        cl::Buffer c(context, CL_MEM_WRITE_ONLY, width*ch*height);
        err |= kernel.setArg(0, a);
        err |= kernel.setArg(1, b);
        err |= kernel.setArg(2, c);

        cl::CommandQueue queue(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE, //0,
            &err);

        // warm up
        for (int i = 0; i < 15; ++i) {
            err |= queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                kernelRangeGlobal,
                kernelRangeLocal,
                NULL,
                NULL);
        }
        cl::finish();

        cl::Event event;
        err |= queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            kernelRangeGlobal,
            kernelRangeLocal,
            NULL,
            &event);
        err |= event.wait();

        // Profiling
        // --------------------------------------------
        cl_ulong start;
        cl_ulong end;
        err |= event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        err |= event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        std::cout
        << "Kernel "
        << kernelName
        << "(): "
        << static_cast<double>(end - start) / 1000.0 / 1000.0
        << " [msec]"
        << std::endl;
    }
    catch (cl::Error &err) {
        std::cerr 
        << "ERROR: "
        << err.what()
        << "("
        << err.err()
        << ")"
        << std::endl;
    }

    return EXIT_SUCCESS;
}
