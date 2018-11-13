
#include <vector>
#include "./corner_pooling-inl.h"

namespace mxnet {
namespace op {



NNVM_REGISTER_OP(CornerPooling)
.set_attr<FCompute>("FCompute<gpu>", CornerPoolingCompute<gpu>);

NNVM_REGISTER_OP(_backward_CornerPooling)
.set_attr<FCompute>("FCompute<gpu>", CornerPoolingGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
