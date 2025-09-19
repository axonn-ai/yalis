#include "nvshmem_comm/nvshmem_comm.h"
#include "nvshmem_comm/coll.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose the Protocol enum
    py::enum_<Protocol>(m, "Protocol")
        .value("SIMPLE", Protocol::SIMPLE)
        .value("LL8", Protocol::LL8);

    py::class_<NVSHMEMCommWrapper, std::shared_ptr<NVSHMEMCommWrapper>>(m, "NVSHMEMCommWrapper")
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, torch::Tensor>())
        .def("destroy", &NVSHMEMCommWrapper::destroy)
        .def("allreduce_preallocated", &NVSHMEMCommWrapper::allreduce_preallocated)
        .def("allocate_tensor", &NVSHMEMCommWrapper::allocate_tensor)
        .def("free_tensor", &NVSHMEMCommWrapper::free_tensor)
        .def("set_kernel_params", &NVSHMEMCommWrapper::set_kernel_params)
        .def("get_rank", &NVSHMEMCommWrapper::get_rank)
        .def("get_world_size", &NVSHMEMCommWrapper::get_world_size)
        .def("get_mype", &NVSHMEMCommWrapper::get_mype)
        .def("get_npes", &NVSHMEMCommWrapper::get_npes)
        .def_static("get_unique_id_bytes", &NVSHMEMCommWrapper::get_unique_id_bytes);
} 