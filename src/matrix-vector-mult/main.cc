#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

int main() {
  std::array<float, 16> mat{};
  std::array<float, 4> vec{};
  std::array<float, 4> result{};
  std::array<float, 4> correct{};

  for (unsigned i = 0; i < mat.size(); ++i) {
    mat[i] = i * 2.0f;
  }

  for (unsigned i = 0; i < 4; i++) {
    vec[i] = i * 3.0f;
    correct[0] += mat[i] * vec[i];
    correct[1] += mat[i + 4] * vec[i];
    correct[2] += mat[i + 8] * vec[i];
    correct[3] += mat[i + 12] * vec[i];
  }

  std::cout << "Matrix:\n{";
  for (float elem : mat)
    std::cout << elem << " ";
  std::cout << "}\n";

  std::cout << "Vector:\n{";
  for (float elem : vec)
    std::cout << elem << " ";
  std::cout << "}\n";

  std::cout << "Correct:\n{";
  for (float elem : correct)
    std::cout << elem << " ";
  std::cout << "}\n";

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform platform = platforms[0];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  assert(!devices.empty());
  cl::Device device = devices[0];

  cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()),
      0 // signals end of property list
  };
  cl::Context context{CL_DEVICE_TYPE_GPU, properties};

  // Read program file
  std::fstream fs{"src/matrix-vector-mult/matvec.cl", std::ifstream::in};
  assert(fs.is_open());
  std::stringstream kernel_ss;
  kernel_ss << fs.rdbuf();
  fs.close();
  std::string kernel_str = kernel_ss.str();

  // Create kernel/queue
  cl::Program program{kernel_str, true};
  cl::Kernel kernel{program, "matvec_mult"};
  cl::CommandQueue queue{context, device};

  // Create buffers
  cl::Buffer mat_buf{context, CL_MEM_READ_ONLY, sizeof(float) * mat.size()};
  cl::Buffer vec_buf{context, CL_MEM_READ_ONLY, sizeof(float) * vec.size()};
  cl::Buffer res_buf{context, CL_MEM_WRITE_ONLY, sizeof(float) * result.size()};

  cl::copy(mat.begin(), mat.end(), mat_buf);
  cl::copy(vec.begin(), vec.end(), vec_buf);

  // Set kernel arguments
  kernel.setArg(0, sizeof(cl::Buffer), &mat_buf);
  kernel.setArg(1, sizeof(cl::Buffer), &vec_buf);
  kernel.setArg(2, sizeof(cl::Buffer), &res_buf);

  // Execute kernel
  cl::NDRange global_range{4};
  cl::NDRange local_range{1};
  cl::EnqueueArgs args{queue, global_range, local_range};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> matvec_mult(
      program, "matvec_mult");
  cl::Event evt = matvec_mult(args, mat_buf, vec_buf, res_buf);
  evt.wait();
  cl::copy(queue, res_buf, result.begin(), result.end());

  std::cout << "Compute result:\n{";
  for (float elem : result)
    std::cout << elem << " ";
  std::cout << "}\n";
}