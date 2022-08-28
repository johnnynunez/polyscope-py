#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"

#include "polyscope/polyscope.h"

#include "polyscope/color_render_image_quantity.h"
#include "polyscope/depth_render_image_quantity.h"
#include "polyscope/floating_quantity_structure.h"
#include "polyscope/scalar_render_image_quantity.h"

#include "utils.h"

namespace py = pybind11;
namespace ps = polyscope;

// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;


void bind_floating_quantities(py::module& m) {

  // == General routines
  m.def("remove_floating_quantity", &ps::removeFloatingQuantity, "Remove any floating quantity");
  m.def("remove_all_floating_quantities", &ps::removeAllFloatingQuantities, "Remove all floating quantities");

  // == Floating scalar images

  bindScalarQuantity<ps::ScalarImageQuantity>(m, "ScalarImageQuantity")
      .def("set_show_fullscreen", &ps::ScalarImageQuantity::setShowFullscreen);

  m.def("add_scalar_image_quantity", &ps::addScalarImageQuantity<Eigen::VectorXd>, "Add scalar image (expects flat)",
        py::return_value_policy::reference);


  // == Floating color images

  bindColorQuantity<ps::ColorImageQuantity>(m, "ColorImageQuantity")
      .def("set_show_fullscreen", &ps::ColorImageQuantity::setShowFullscreen);

  m.def("add_color_image_quantity", &ps::addColorImageQuantity<Eigen::MatrixXd>, "Add color image (expects flatx3)",
        py::return_value_policy::reference);

  // == Floating render depth image

  py::class_<ps::DepthRenderImageQuantity>(m, "DepthRenderImageQuantity")
    .def("set_enabled", &ps::DepthRenderImageQuantity::setEnabled, "Enable the image")
    .def("set_material", &ps::DepthRenderImageQuantity::setMaterial, "Set material")
    .def("get_material", &ps::DepthRenderImageQuantity::getMaterial, "Get material")
    .def("set_transparency", &ps::DepthRenderImageQuantity::setTransparency, "Set transparency")
    .def("get_transparency", &ps::DepthRenderImageQuantity::getTransparency, "Get transparency")
    .def("set_color", &ps::DepthRenderImageQuantity::setColor, "Set color")
    .def("get_color", &ps::DepthRenderImageQuantity::getColor, "Get color")
  ;
  
  m.def("add_depth_render_image_quantity", &ps::addDepthRenderImageQuantity<Eigen::VectorXd, Eigen::MatrixXd>, "Add depth render image (expects flat)",
        py::return_value_policy::reference);


  py::class_<ps::ColorRenderImageQuantity>(m, "ColorRenderImageQuantity")
    .def("set_enabled", &ps::ColorRenderImageQuantity::setEnabled, "Enable the image")
    .def("set_material", &ps::ColorRenderImageQuantity::setMaterial, "Set material")
    .def("get_material", &ps::ColorRenderImageQuantity::getMaterial, "Get material")
    .def("set_transparency", &ps::ColorRenderImageQuantity::setTransparency, "Set transparency")
    .def("get_transparency", &ps::ColorRenderImageQuantity::getTransparency, "Get transparency")
  ;
  
  m.def("add_color_render_image_quantity", &ps::addColorRenderImageQuantity<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>, "Add color render image (expects flat)",
        py::return_value_policy::reference);


  bindScalarQuantity<ps::ScalarRenderImageQuantity>(m, "ScalarRenderImageQuantity") 
    .def("set_material", &ps::ScalarRenderImageQuantity::setMaterial, "Set material")
    .def("get_material", &ps::ScalarRenderImageQuantity::getMaterial, "Get material")
    .def("set_transparency", &ps::ScalarRenderImageQuantity::setTransparency, "Set transparency")
    .def("get_transparency", &ps::ScalarRenderImageQuantity::getTransparency, "Get transparency")
  ;
  
  m.def("add_scalar_render_image_quantity", &ps::addScalarRenderImageQuantity<Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>, "Add scalar render image (expects flat)",
        py::return_value_policy::reference);

  // bindStructure<ps::FloatingQuantityStructure>(m, "FloatingQuantityStructure");
}
