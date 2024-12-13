// Copyright 2023 Nesterov Alexander
#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void gordeeva_t_shell_sort_batcher_merge_mpi::shellSort(std::vector<int>& arr) {
  size_t arr_length = arr.size();
  for (size_t step = arr_length / 2; step > 0; step /= 2) {
    for (size_t i = step; i < arr_length; i++) {
      size_t j = i;
      while (j >= step && arr[j - step] > arr[j]) {
        std::swap(arr[j], arr[j - step]);
        j -= step;
      }
    }
  }
}

std::vector<int> gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(int size, int down, int upp) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(down, upp);

  std::vector<int> v(size);
  for (auto& number : v) {
    number = dis(gen);
  }
  return v;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::pre_processing() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    throw std::runtime_error("Invalid input data in pre_processing.");
  }

  size_t sz = taskData->inputs_count[0];
  auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.assign(input_tmp, input_tmp + sz);
  res_.resize(sz);

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::validation() {
  if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
  if (taskData->inputs_count[0] <= 0) return false;
  if (taskData->outputs_count.size() != 1) return false;
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::run() {
  shellSort(input_);
  res_ = input_;
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::post_processing() {
  if (taskData->outputs.empty()) {
    throw std::runtime_error("Invalid output data in post_processing.");
  }

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_matr);
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::pre_processing() {
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
      throw std::runtime_error("Invalid input data in pre_processing.");
    }

    sz_mpi = taskData->inputs_count[0];
    auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    input_.assign(input_tmp, input_tmp + sz_mpi);
    res_.resize(sz_mpi);
  }
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::validation() {
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
    if (taskData->inputs_count[0] <= 0) return false;
    if (taskData->outputs_count.size() != 1) return false;
  }
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::run() {
  size_t rank = world.rank();
  size_t size = world.size();

  boost::mpi::broadcast(world, sz_mpi, 0);

  size_t delta = sz_mpi / size;
  size_t ost = sz_mpi % size;

  std::vector<int> sz_rk(size, static_cast<int>(delta));

  std::cout << "[ MY-4 ]" << ost << std::endl;
  for (size_t i = 0; i < ost; ++i) {
    sz_rk[i]++;

    std::cout << world.rank() << ": " << sz_rk[i] << std::endl;
  }

  boost::mpi::broadcast(world, input_.data(), input_.size(), 0);

  local_input_.resize(sz_rk[rank]);
  boost::mpi::scatterv(world, input_, sz_rk, local_input_.data(), 0);

  world.barrier();
  shellSort(local_input_);

  for (size_t i = 0; i < size; ++i) {
    if (rank % 2 == i % 2 && rank + 1 < size) {
      batcher_merge(rank, rank + 1);
    } else if (rank % 2 != i % 2 && rank > 0) {
      batcher_merge(rank - 1, rank);
    }
  }

  std::cout << "Rank " << rank << " local_input size: " << local_input_.size() << std::endl;
  if (rank == 0) {
    std::cout << "Rank " << rank << " res_ size: " << res_.size() << std::endl;
    for (size_t i = 0; i < sz_rk.size(); ++i) {
      std::cout << "Expected size for rank " << i << ": " << sz_rk[i] << std::endl;
    }
  }

  boost::mpi::gatherv(world, local_input_.data(), local_input_.size(), res_.data(), sz_rk, 0);

  std::cout << "[ MY5 ]" << world.rank() << std::endl;
  return true;
}

void gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::batcher_merge(size_t rank1, size_t rank2) {
  int rank = world.rank();
  std::vector<int> arr;

  if (rank == rank1) {
    world.send(static_cast<int>(rank2), 0, local_input_);
    world.recv(static_cast<int>(rank2), 0, arr);
  } else if (rank == rank2) {
    world.recv(static_cast<int>(rank1), 0, arr);
    world.send(static_cast<int>(rank1), 0, local_input_);
  }

  if (!arr.empty()) {
    std::vector<int> merge_arr(local_input_.size() + arr.size());
    std::merge(local_input_.begin(), local_input_.end(), arr.begin(), arr.end(), merge_arr.begin());

    if (rank == rank1) {
      local_input_.assign(merge_arr.begin(), merge_arr.begin() + local_input_.size());
    } else {
      local_input_.assign(merge_arr.end() - local_input_.size(), merge_arr.end());
    }
  }
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::post_processing() {
  if (world.rank() == 0 && !taskData->outputs.empty()) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
