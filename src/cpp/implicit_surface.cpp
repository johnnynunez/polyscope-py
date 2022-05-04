#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Eigen/Dense"

#include "polyscope/implicit_surface.h"
#include "polyscope/polyscope.h"

#include "utils.h"

namespace py = pybind11;
namespace ps = polyscope;

// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// clang-format off
void bind_implicit_surface(py::module& m) {

  // Bind the options object and associated enum
  
  py::enum_<ps::ImplicitRenderMode>(m, "ImplicitRenderMode")
    .value("sphere_march", ps::ImplicitRenderMode::SphereMarch)
    .value("fixed_step", ps::ImplicitRenderMode::FixedStep)
    .export_values(); 

  py::class_<ps::ImplicitRenderOpts>(m, "ImplicitRenderOpts")
    .def(py::init<>())
    .def_readwrite("mode", &ps::ImplicitRenderOpts::mode)
    .def_readwrite("step_factor", &ps::ImplicitRenderOpts::stepFactor)
    .def_readwrite("normal_sample_eps", &ps::ImplicitRenderOpts::normalSampleEps)
    .def_readwrite("n_max_steps", &ps::ImplicitRenderOpts::nMaxSteps)
    .def_readwrite("subsample_factor", &ps::ImplicitRenderOpts::subsampleFactor)
    .def("set_miss_dist", [](ps::ImplicitRenderOpts &o, float newVal, bool isRelative) { 
          o.missDist = ps::ScaledValue<float>(newVal, isRelative);
        })
    .def("set_hit_dist", [](ps::ImplicitRenderOpts &o, float newVal, bool isRelative) { 
          o.hitDist = ps::ScaledValue<float>(newVal, isRelative);
        })
    .def("set_step_size", [](ps::ImplicitRenderOpts &o, float newVal, bool isRelative) { 
          o.stepSize = ps::ScaledValue<float>(newVal, isRelative);
        })
  ;

  // == Implicit render functions

  m.def("render_implicit_surface", 
        [](std::string name, const std::function<Eigen::VectorXf(Eigen::MatrixXf)>& func, ps::ImplicitRenderOpts opts) {

          // does some copy conversions to/from Eigen types
          auto helperFuncConvertBatchTypes = [&](const std::vector<glm::vec3>& arr) {
            Eigen::MatrixXf arrEigen(arr.size(), 3);
            for(size_t i = 0; i < arr.size(); i++) {
              for(size_t j = 0; j < 3; j++) {
                arrEigen(i,j) = arr[i][j];
              }
            }
            Eigen::VectorXf values = func(arrEigen);
            std::vector<float> stdValues(values.size());
            for(size_t i = 0; i < values.size(); i++) { 
              stdValues[i] = values[i];
            }
            return stdValues;
          };

          return ps::renderImplicitSurfaceBatch(name, helperFuncConvertBatchTypes, opts);
      
        }, py::arg("name"), py::arg("func"), py::arg("opts"), py::return_value_policy::reference, "Render an implicit surface");


  m.def("render_implicit_surface_color", 
        [](std::string name, 
          const std::function<Eigen::VectorXf(Eigen::MatrixXf)>& func, 
          const std::function<Eigen::MatrixXf(Eigen::MatrixXf)>& funcColor, 
          ps::ImplicitRenderOpts opts) {

          // does some copy conversions to/from Eigen types
          auto helperFuncConvertBatchTypes = [&](const std::vector<glm::vec3>& arr) {
            Eigen::MatrixXf arrEigen(arr.size(), 3);
            for(size_t i = 0; i < arr.size(); i++) {
              for(size_t j = 0; j < 3; j++) {
                arrEigen(i,j) = arr[i][j];
              }
            }
            Eigen::VectorXf values = func(arrEigen);
            std::vector<float> stdValues(values.size());
            for(size_t i = 0; i < values.size(); i++) { 
              stdValues[i] = values[i];
            }
            return stdValues;
          };
          
          auto helperFuncConvertBatchTypesColor = [&](const std::vector<glm::vec3>& arr) {
            Eigen::MatrixXf arrEigen(arr.size(), 3);
            for(size_t i = 0; i < arr.size(); i++) {
              for(size_t j = 0; j < 3; j++) {
                arrEigen(i,j) = arr[i][j];
              }
            }
            Eigen::MatrixXf values = funcColor(arrEigen);
            std::vector<glm::vec3> stdValues(values.rows());
            for(size_t i = 0; i < values.rows(); i++) { 
              for(size_t j = 0; j < 3; j++) {
                stdValues[i][j] = values(i,j);
              }
            }
            return stdValues;
          };

          return ps::renderImplicitSurfaceColorBatch(name, helperFuncConvertBatchTypes, helperFuncConvertBatchTypesColor, opts);
      
        }, py::arg("name"), py::arg("func"), py::arg("func_color"), py::arg("opts"), py::return_value_policy::reference, "Render an implicit surface with color");
  

  m.def("render_implicit_surface_scalar", 
        [](std::string name, 
          const std::function<Eigen::VectorXf(Eigen::MatrixXf)>& func, 
          const std::function<Eigen::VectorXd(Eigen::MatrixXf)>& funcScalar, 
          ps::ImplicitRenderOpts opts, ps::DataType dataType) {

          // does some copy conversions to/from Eigen types
          auto helperFuncConvertBatchTypes = [&](const std::vector<glm::vec3>& arr) {
            Eigen::MatrixXf arrEigen(arr.size(), 3);
            for(size_t i = 0; i < arr.size(); i++) {
              for(size_t j = 0; j < 3; j++) {
                arrEigen(i,j) = arr[i][j];
              }
            }
            Eigen::VectorXf values = func(arrEigen);
            std::vector<float> stdValues(values.size());
            for(size_t i = 0; i < values.size(); i++) { 
              stdValues[i] = values[i];
            }
            return stdValues;
          };
          
          auto helperFuncConvertBatchTypesScalar = [&](const std::vector<glm::vec3>& arr) {
            Eigen::MatrixXf arrEigen(arr.size(), 3);
            for(size_t i = 0; i < arr.size(); i++) {
              for(size_t j = 0; j < 3; j++) {
                arrEigen(i,j) = arr[i][j];
              }
            }
            Eigen::VectorXd values = funcScalar(arrEigen);
            std::vector<double> stdValues(values.size());
            for(size_t i = 0; i < values.size(); i++) { 
              stdValues[i] = values[i];
            }
            return stdValues;
          };

          return ps::renderImplicitSurfaceScalarBatch(name, helperFuncConvertBatchTypes, helperFuncConvertBatchTypesScalar, opts, dataType);
      
        }, py::arg("name"), py::arg("func"), py::arg("funcScalar"), py::arg("opts"), py::arg("data_type")=ps::DataType::STANDARD, py::return_value_policy::reference, "Render an implicit surface with a scalar field on it");

}
