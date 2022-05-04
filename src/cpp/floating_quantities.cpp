#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"

#include "polyscope/polyscope.h"

#include "polyscope/color_render_image.h"
#include "polyscope/depth_render_image.h"
#include "polyscope/floating_quantity_structure.h"
#include "polyscope/scalar_render_image.h"

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

  bindScalarQuantity<ps::FloatingScalarImageQuantity>(m, "FloatingScalarImageQuantity")
      .def("set_show_fullscreen", &ps::FloatingScalarImageQuantity::setShowFullscreen);

  m.def("add_floating_scalar_image", &ps::addFloatingScalarImage<Eigen::VectorXd>, "Add scalar image (expects flat)",
        py::return_value_policy::reference);


  // == Floating color images

  bindColorQuantity<ps::FloatingColorImageQuantity>(m, "FloatingColorImageQuantity")
      .def("set_show_fullscreen", &ps::FloatingColorImageQuantity::setShowFullscreen);

  m.def("add_floating_color_image", &ps::addFloatingColorImage<Eigen::MatrixXd>, "Add color image (expects flatx3)",
        py::return_value_policy::reference);

  // == Floating render depth image

  py::class_<ps::DepthRenderImage>(m, "DepthRenderImage")
    .def("set_enabled", &ps::DepthRenderImage::setEnabled, "Enable the image")
    .def("set_material", &ps::DepthRenderImage::setMaterial, "Set material")
    .def("get_material", &ps::DepthRenderImage::getMaterial, "Get material")
    .def("set_transparency", &ps::DepthRenderImage::setTransparency, "Set transparency")
    .def("get_transparency", &ps::DepthRenderImage::getTransparency, "Get transparency")
    .def("set_color", &ps::DepthRenderImage::setColor, "Set color")
    .def("get_color", &ps::DepthRenderImage::getColor, "Get color")
  ;
  

  py::class_<ps::ColorRenderImage>(m, "ColorRenderImage")
    .def("set_enabled", &ps::ColorRenderImage::setEnabled, "Enable the image")
    .def("set_material", &ps::ColorRenderImage::setMaterial, "Set material")
    .def("get_material", &ps::ColorRenderImage::getMaterial, "Get material")
    .def("set_transparency", &ps::ColorRenderImage::setTransparency, "Set transparency")
    .def("get_transparency", &ps::ColorRenderImage::getTransparency, "Get transparency")
  ;

  bindScalarQuantity<ps::ScalarRenderImage>(m, "ScalarRenderImage") 
    .def("set_material", &ps::ScalarRenderImage::setMaterial, "Set material")
    .def("get_material", &ps::ScalarRenderImage::getMaterial, "Get material")
    .def("set_transparency", &ps::ScalarRenderImage::setTransparency, "Set transparency")
    .def("get_transparency", &ps::ScalarRenderImage::getTransparency, "Get transparency")
  ;

  // bindStructure<ps::FloatingQuantityStructure>(m, "FloatingQuantityStructure");
}
