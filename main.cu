#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <fmt/core.h>

#include "jitify.hpp"

// UDF example
// Usage

int main() {
  using clk = std::chrono::system_clock;
  using ms = std::chrono::milliseconds;
  using us = std::chrono::microseconds;
  using ns = std::chrono::nanoseconds;
  using s = std::chrono::seconds;
  using time_point_t = decltype(clk::now());

  auto const program_source_1 = std::string{
      "my_program\n"
      "__global__\n"
      "void my_kernel(int *data, unsigned int num) {\n"
      "  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (id < num) {\n"
      "    int row = data[id];\n"
  };
  auto program_source_2 = std::string{};
  const auto program_source_3 = std::string{
      "    data[id] = row;\n"
      "  }\n"
      "}\n"
  };

  // read file into program_source_2
  auto file = std::fstream("UDF");
  auto buffer = std::stringstream{};
  buffer << file.rdbuf();
  program_source_2 = buffer.str();

  auto program_source = program_source_1 + program_source_2 + "\n" + program_source_3;
  fmt::print("[generated codes]\n{}", program_source);

  auto before = clk::now();
  static auto kernel_cache = jitify::JitCache{};
  auto program = kernel_cache.program(program_source);
  auto after = clk::now();
  auto duration = after - before;
  auto n_us = std::chrono::duration_cast<us>(duration).count();
  fmt::print("[compile time] {} us\n", n_us);

  int len = 20;
  int *h_data = new int[len];
  for (int i = 0; i < len; ++i) h_data[i] = i;

  fmt::print("[input items]\n");
  for (int i = 0; i < len; ++i) fmt::print("{} ", h_data[i]);
  fmt::print("\n");

  int *d_data;
  cudaMalloc((void **) &d_data, sizeof(int) * len);
  cudaMemcpy(d_data, h_data, sizeof(int) * len, cudaMemcpyHostToDevice);
  dim3 grid(5);
  dim3 block(4);
  using jitify::reflection::type_of;
  program.kernel("my_kernel")
      .instantiate()
      .configure(grid, block)
      .launch(d_data, len);
  cudaMemcpy(h_data, d_data, sizeof(int) * len, cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  fmt::print("[output items]\n");
  for (int i = 0; i < len; ++i) fmt::print("{} ", h_data[i]);
  fmt::print("\n");

  delete[] h_data;

}
