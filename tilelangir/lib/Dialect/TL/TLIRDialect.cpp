//===- TLIRDialect.cpp - TLIR dialect registration ----------------------===//

#include "tilelangir/Dialect/TL/TLIROps.h"

using namespace mlir;
using namespace mlir::tlir;

#include "tilelangir/Dialect/TL/TLIRDialect.cpp.inc"

#include "tilelangir/Dialect/TL/TLIRAttrs.enum.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tilelangir/Dialect/TL/TLIRAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "tilelangir/Dialect/TL/TLIROps.cpp.inc"

void TLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tilelangir/Dialect/TL/TLIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tilelangir/Dialect/TL/TLIRAttrs.cpp.inc"
      >();
}
