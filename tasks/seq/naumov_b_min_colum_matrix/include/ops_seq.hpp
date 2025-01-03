#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_min_colum_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res;
};

}  // namespace naumov_b_min_colum_matrix_seq