#include <tvm/ir/attrs.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace transform {

class DeclBufferShapeCollector : public StmtVisitor {
 public:
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> handle_to_buffer;

  void VisitStmt_(const DeclBufferNode* op) final {
    handle_to_buffer[op->buffer->data] = op->buffer;
    StmtVisitor::VisitStmt_(op);
  }

  static std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> Collect(
      const PrimFunc& func) {
    DeclBufferShapeCollector v;
    v(func->body);
    return v.handle_to_buffer;
  }
};

Pass AnnotateBufferInfoPass() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    IRModule updated_mod = mod->ShallowCopy();

    static std::unordered_map<std::string, int> dtype_map = {
        {"uint8", 0},   {"uint16", 1},  {"uint32", 2},   {"uint64", 3},
        {"int8", 4},    {"int16", 5},   {"int32", 6},    {"int64", 7},
        {"float16", 8}, {"float32", 9}, {"float64", 10}, {"handle", 11},
    };
    const int handle_base = dtype_map["handle"];

    for (const auto& [gvar, base_func] : mod->functions) {
      auto opt = base_func.as<PrimFunc>();
      if (!opt) continue;

      PrimFunc func = opt.value();
      Optional<Target> tgt = func->GetAttr<Target>("target");
      if (!tgt.defined()) continue;

      std::string kind = tgt.value()->kind->name;

      Array<Array<Integer>> buffer_shapes;
      Array<Integer> arg_kinds;

      tvm::Map<Var, Integer> shape_var_indices;
      for (size_t i = 0; i < func->params.size(); ++i) {
        shape_var_indices.Set(func->params[i], Integer(-static_cast<int>(i)));
      }

      // 收集 decl_buffer 中的 buffer 映射
      auto decl_buffers = DeclBufferShapeCollector::Collect(func);

      for (const Var& param : func->params) {
        DataType param_dtype = param->dtype;
        std::string dtype_str = tvm::runtime::DLDataType2String(DLDataType(param_dtype));

        Buffer buf;
        if (func->buffer_map.count(param)) {
          buf = func->buffer_map.at(param);
        } else if (decl_buffers.count(param)) {
          buf = decl_buffers.at(param);
        }

        if (buf.defined()) {
          std::string buf_dtype = tvm::runtime::DLDataType2String(DLDataType(buf->dtype));
          int type_code = handle_base + dtype_map.at(buf_dtype);
          arg_kinds.push_back(IntImm(DataType::Int(32), type_code));

          Array<Integer> shape_repr;
          shape_repr.push_back(IntImm(DataType::Int(32), buf->dtype.bits() / 8));
          for (PrimExpr dim : buf->shape) {
            if (const auto* imm = dim.as<IntImmNode>()) {
              shape_repr.push_back(IntImm(DataType::Int(32), imm->value));
            } else if (const auto* var = dim.as<VarNode>()) {
              Var dim_var = GetRef<Var>(var);
              if (shape_var_indices.count(dim_var)) {
                shape_repr.push_back(shape_var_indices.at(dim_var));
              } else {
                LOG(WARNING) << "Unknown shape var: " << dim_var;
                shape_repr.push_back(IntImm(DataType::Int(32), -999));
              }
            } else {
              LOG(WARNING) << "Unsupported dim expr: " << dim;
              shape_repr.push_back(IntImm(DataType::Int(32), -998));
            }
          }
          buffer_shapes.push_back(shape_repr);
        } else {
          int type_code = dtype_map.at(dtype_str);
          arg_kinds.push_back(IntImm(DataType::Int(32), type_code));
          buffer_shapes.push_back(
              {IntImm(DataType::Int(32), param_dtype.bits() / 8), IntImm(DataType::Int(32), 1)});
        }
      }

      func = WithAttrs(std::move(func), {{tvm::attr::kP2Psizes, buffer_shapes}});
      func = WithAttrs(std::move(func), {{tvm::attr::kP2PArgTypes, arg_kinds}});
      updated_mod->Add(gvar, func);
    }

    return updated_mod;
  };

  return tvm::transform::CreateModulePass(pass_func,
                                          /*opt_level=*/0, "tir.AnnotateBufferInfo", {});
}

TVM_REGISTER_GLOBAL("tir.transform.AnnotateBufferInfo").set_body_typed(AnnotateBufferInfoPass);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
