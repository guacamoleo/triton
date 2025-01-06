#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#define DEBUG_TYPE "tritonamdgpu-tile-tensors"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

//
class TensorTiler {
  int maxTileSizeM;
  int maxTileSizeN;
  int tileSizeM;
  int tileSizeN;

  public:
  TensorTiler(Operation *zeroInitOp, int maxM, int maxN) {

  }
  LogicalResult tileTensor();
};

LogicalResult TensorTiler::tileTensor() {

}

///////////////////////////////////////////////////////////////////////////////
// Coppied from OptimizeAccumulatorInit.cpp


bool isConstantZeroTensor(Value v) {
  return (matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat()));
}

std::pair<Value, Operation *> getAccumulatorUseAndDef(Operation *op) {
  assert(op->hasTrait<OpTrait::DotLike>() && "Expected a dot-like operation");
  if (auto wgDotOp = dyn_cast<triton::nvidia_gpu::WarpGroupDotOp>(op)) {
    return std::make_pair(wgDotOp.getC(), wgDotOp);
  }
  return std::make_pair(nullptr, nullptr);
}

std::optional<std::pair<Operation *, int>> findZeroInitOp(Value accUse,
                                                          Operation *accDef,
                                                          scf::ForOp forOp,
                                                          bool &loopArgIsZero) {
  Value v = accUse;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    assert(arg.getOwner() == forOp.getBody());
    if (isConstantZeroTensor(forOp.getInitArgs()[arg.getArgNumber() - 1])) {
      // 2.
      loopArgIsZero = true;
    }
    v = forOp.getBody()->getTerminator()->getOperand(arg.getArgNumber() - 1);
  }

  auto defOp = v.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  if (auto selOp = dyn_cast<arith::SelectOp>(defOp)) {
    if (!selOp.getCondition().getType().isInteger(1))
      return std::nullopt;
    if (isConstantZeroTensor(selOp.getTrueValue()) ||
        isConstantZeroTensor(selOp.getFalseValue())) {
      // 3. select op
      return std::make_pair(selOp, 0);
    }
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    unsigned resultIndex = 0;
    for (; resultIndex < ifOp.getNumResults(); ++resultIndex) {
      if (ifOp.getResult(resultIndex) == v)
        break;
    }
    Value thenVal = ifOp.thenYield()->getOperand(resultIndex);
    Value elseVal = ifOp.elseYield()->getOperand(resultIndex);
    if (isConstantZeroTensor(thenVal) || isConstantZeroTensor(elseVal)) {
      // Make sure that the other value is not defined in the if itself, but
      // passed from outside
      if (thenVal.getParentBlock()->getParentOp() == ifOp ||
          elseVal.getParentBlock()->getParentOp() == ifOp) {
        return std::nullopt;
      }
      // 3. if op
      return std::make_pair(ifOp, resultIndex);
    }
  }
  return std::nullopt;
}

class TritonAMDGPUTileTensorPass
    : public TritonAMDGPUTileTensorBase<TritonAMDTileTensorPass> {
public:
  TritonAMDGPUTileTensorPass() = default;
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<Operation *> mmaOps;
    // Get mma ops.
    m.walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::DotLike>() && dotSupportsAccInitFlag(op)) {
        mmaOps.push_back(op);
      }
    });

    
    // For each mma op, find where the accumulator is initialized with zero
    // It can be:
    // 1. A constant zero
    // 2. Initialized with zero as the loop argument
    // 3. Initialized with zero in the if op or with a select op in current
    //   or any of the previous loop iterations
    SmallVector<std::pair<Operation *, int>> zeroInitOps;
    for (Operation *mmaOp : mmaOps) {
      Location loc = mmaOp->getLoc();

      scf::ForOp forOp = dyn_cast<scf::ForOp>(mmaOp->getParentOp());
      if (!forOp) {
        continue;
      }

      // Find the accumulator
      auto [accUse, accDef] = getAccumulatorUseAndDef(mmaOp);
      if (!accUse || !accDef) {
        continue;
      }
      if (isConstantZeroTensor(accUse)) {
        // 1. constant
        zeroInitOps.push_back(std::make_pair(accUse, 0));
        continue;
      }

      bool loopArgIsZero = false;
      std::optional<std::pair<Operation *, int>> zeroInitOp =
          findZeroInitOp(accUse, accDef, forOp, loopArgIsZero);
      if (!zeroInitOp) {
        continue;
      }
      zeroInitOps.push_back(zeroInitOp);
    }
    int maxM = 32;
    int maxN = 32;

    for (std::pair<Operation *, int> zeroInitOp : zeroInitOps) {
      Operation *op = zeroInitOp->first;
      DBGS() << "ZeroOp: " << *op << '\n';
      // TODO - may want an analysis pass here to calculate good tile size.
      TensorTiler tensorTiler(zeroInitOp, maxM, maxN);
      tensorTiler.tileTensor(); // may fail
    }
  }

};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUTileTensorPass() {
  return std::make_unique<TritonAMDGPUTileTensorPass>();
}
