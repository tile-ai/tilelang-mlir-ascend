// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/ascend910/op/builtin.h
 * \brief Compatibility shim for the Ascend 910 code generators.
 *
 * The legacy (v0.1.4 based) tree shipped its own ``op/builtin.h`` with the
 * builtin declarations plus the ``CHECK`` assertion macro of that TVM version.
 * On v0.1.15 the builtins live in ``tvm/tirx/builtin.h`` and the assertion
 * macro is ``ICHECK``; this header keeps the code generators unchanged.
 */

#ifndef TILELANG_ASCEND910_OP_BUILTIN_H_
#define TILELANG_ASCEND910_OP_BUILTIN_H_

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

// Some MLIR/CANN headers define a variadic CHECK macro; the legacy code
// generators use the single-condition TVM form.
#ifdef CHECK
#undef CHECK
#endif
#define CHECK(condition) ICHECK(condition)

#endif // TILELANG_ASCEND910_OP_BUILTIN_H_