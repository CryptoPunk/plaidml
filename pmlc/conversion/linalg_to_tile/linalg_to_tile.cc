// Copyright 2021, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/conversion/linalg_to_tile/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

#include "pmlc/util/ident.h"

namespace pmlc::conversion::linalg_to_tile {

namespace layer = dialect::layer;
namespace tile = dialect::tile;

using namespace mlir;         // NOLINT
using namespace mlir::linalg; // NOLINT

using util::AggregationKind;
using util::CombinationKind;

namespace {

struct Matcher {
  LogicalResult operator()(Operation *op) { return success(match(op)); }
  virtual bool match(Operation *op) const { return false; }
};

struct AlwaysTrue : Matcher {
  bool match(Operation *op) const final { return true; }
};

struct ConvOpConversion : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct CopyOpConversion : public OpConversionPattern<CopyOp> {
  using OpConversionPattern<CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct FillOpConversion : public OpConversionPattern<FillOp> {
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FillOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct GenericOpConversion : public OpConversionPattern<GenericOp> {
  using OpConversionPattern<GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct IndexOpConversion : public OpConversionPattern<IndexOp> {
  using OpConversionPattern<IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct InitTensorOpConversion : public OpConversionPattern<InitTensorOp> {
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct PadTensorOpConversion : public OpConversionPattern<PadTensorOp> {
  using OpConversionPattern<PadTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PadTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct RangeOpConversion : public OpConversionPattern<RangeOp> {
  using OpConversionPattern<RangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RangeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct TensorCollapseShapeOpConversion
    : public OpConversionPattern<TensorCollapseShapeOp> {
  using OpConversionPattern<TensorCollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorCollapseShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct TensorExpandShapeOpConversion
    : public OpConversionPattern<TensorExpandShapeOp> {
  using OpConversionPattern<TensorExpandShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorExpandShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct TiledLoopOpConversion : public OpConversionPattern<TiledLoopOp> {
  using OpConversionPattern<TiledLoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TiledLoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<YieldOp> {
  using OpConversionPattern<YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

template <typename FromOpType, typename Matcher = AlwaysTrue>
struct PoolingOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FromOpType op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

template <typename FromOpType, typename Matcher = AlwaysTrue>
struct ContractionOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FromOpType op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO
    return success();
  }
};

struct LowerLinalgToTilePass
    : public LowerLinalgToTileBase<LowerLinalgToTilePass> {
  void runOnOperation() final {
    // Set up target (i.e. what is legal)
    ConversionTarget target(getContext());
    LinalgToTileTypeConverter converter;
    target.addLegalDialect<mlir::StandardOpsDialect, //
                           mlir::math::MathDialect,  //
                           layer::LayerDialect,      //
                           tile::TileDialect>();
    target.addLegalOp<scf::ForOp,   //
                      scf::YieldOp, //
                      scf::IfOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    // Setup rewrite patterns
    RewritePatternSet patterns(&getContext());
    patterns
        .insert<ConvOpConversion,                                            //
                CopyOpConversion,                                            //
                FillOpConversion,                                            //
                FuncOpConversion,                                            //
                GenericOpConversion,                                         //
                IndexOpConversion,                                           //
                InitTensorOpConversion,                                      //
                PadTensorOpConversion,                                       //
                RangeOpConversion,                                           //
                TensorCollapseShapeOpConversion,                             //
                TensorExpandShapeOpConversion,                               //
                TiledLoopOpConversion,                                       //
                YieldOpConversion,                                           //
                PoolingOpConversion<PoolingMaxOp>,                           //
                PoolingOpConversion<PoolingMinOp>,                           //
                PoolingOpConversion<PoolingSumOp>,                           //
                ContractionOpConversion<MatmulColumnMajorOp>,                //
                ContractionOpConversion<MatmulI8I8I32Op>,                    //
                ContractionOpConversion<MatmulI16I16I32Op>,                  //
                ContractionOpConversion<MatmulI32I32I32Op>,                  //
                ContractionOpConversion<MatvecI8I8I32Op>,                    //
                ContractionOpConversion<MatvecI16I16I32Op>,                  //
                ContractionOpConversion<MatvecI32I32I32Op>,                  //
                ContractionOpConversion<VecmatI8I8I32Op>,                    //
                ContractionOpConversion<VecmatI16I16I32Op>,                  //
                ContractionOpConversion<VecmatI32I32I32Op>,                  //
                ContractionOpConversion<DotI8I8I32Op>,                       //
                ContractionOpConversion<DotI16I16I32Op>,                     //
                ContractionOpConversion<DotI32I32I32Op>,                     //
                ContractionOpConversion<BatchMatmulI8I8I32Op>,               //
                ContractionOpConversion<BatchMatmulI16I16I32Op>,             //
                ContractionOpConversion<BatchMatmulI32I32I32Op>,             //
                ContractionOpConversion<ConvWOp>,                            //
                ContractionOpConversion<ConvNWCOp>,                          //
                ContractionOpConversion<ConvNCWOp>,                          //
                ContractionOpConversion<ConvHWOp>,                           //
                ContractionOpConversion<ConvNHWCOp>,                         //
                ContractionOpConversion<ConvNCHWOp>,                         //
                ContractionOpConversion<ConvDHWOp>,                          //
                ContractionOpConversion<ConvNDHWCOp>,                        //
                ContractionOpConversion<ConvNCDHWOp>,                        //
                ContractionOpConversion<DepthwiseConvInputNHWCFilterHWCFOp>, //
                ContractionOpConversion<DepthwiseConvInputNHWCFilterHWCOp>,  //
                ContractionOpConversion<ConvInputNWCFilterWCFOp>,            //
                ContractionOpConversion<ConvInputNCWFilterWCFOp>,            //
                ContractionOpConversion<ConvInputNHWCFilterHWCFOp>,          //
                ContractionOpConversion<ConvInputNCHWFilterHWCFOp>,          //
                ContractionOpConversion<ConvInputNDHWCFilterDHWCFOp>,        //
                ContractionOpConversion<ConvInputNCDHWFilterDHWCFOp>,        //
                ContractionOpConversion<PoolingNHWCSumFOp>,                  //
                ContractionOpConversion<PoolingNHWCMaxI8Op>,                 //
                ContractionOpConversion<PoolingNHWCMaxI16Op>,                //
                ContractionOpConversion<PoolingNHWCMaxI32Op>,                //
                ContractionOpConversion<PoolingNHWCMaxFOp>,                  //
                ContractionOpConversion<PoolingNHWCMinFOp>>(&getContext());

    populateLinalgToTileSpecialPatterns(patterns);

    // Run the conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerLinalgToTilePass() {
  return std::make_unique<LowerLinalgToTilePass>();
}

} // namespace pmlc::conversion::linalg_to_tile
