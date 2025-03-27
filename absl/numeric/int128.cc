// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/numeric/int128.h"

#include <stddef.h>

#include <cassert>
#include <iomanip>
#include <ostream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>
#include <type_traits>

#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"

#if defined(_MSC_VER) && defined(_M_X64) && !defined(_M_ARM64EC) && \
    _MSC_VER >= 1920
// The _udiv128 intrinsic is available starting in Visual Studio 2019 RTM.
// https://learn.microsoft.com/en-us/cpp/intrinsics/udiv128?view=msvc-170
#include <immintrin.h>
#pragma intrinsic(_udiv128)
#endif

namespace absl {
ABSL_NAMESPACE_BEGIN

namespace {

// If the result of dividing a uint128 by a uint64 fits within 64 bits,
// the division can be implemented efficiently using an intrinsic instruction.
// If the result does not fit within 64 bits, the behavior is undefined.
inline void DivModImpl(uint128 dividend, uint64_t divisor,
                       uint64_t* quotient_ret, uint64_t* remainder_ret) {
  uint64_t high = Uint128High64(dividend);
  uint64_t low = Uint128Low64(dividend);

  assert(divisor != 0 && high < divisor);

  if (high == 0) {
    *quotient_ret = low / divisor;
    *remainder_ret = low % divisor;
    return;
  }
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
  uint64_t qt, rem;
  __asm__ __volatile__("divq %[v]"
                       : "=a"(qt), "=d"(rem)
                       : [v] "r"(divisor), "a"(low), "d"(high));
  *quotient_ret = qt;
  *remainder_ret = rem;
#elif defined(_MSC_VER) && defined(_M_X64) && !defined(_M_ARM64EC) && \
    _MSC_VER >= 1920
  *quotient_ret = _udiv128(high, low, divisor, remainder_ret);
#elif defined(ABSL_HAVE_INTRINSIC_INT128)
  uint64_t result =
      Uint128Low64(static_cast<unsigned __int128>(dividend) / divisor);
  *quotient_ret = result;
  *remainder_ret = low - result * divisor;
#else
  if (divisor <= 0xffffffff) {
    uint64_t d1 = (high << 32) | (low >> 32);
    uint64_t qt = d1 / divisor;
    uint64_t rem = d1 % divisor;
    uint64_t d2 = (rem << 32) | (low & 0xffffffff);
    *quotient_ret = (qt << 32) | (d2 / divisor);
    *remainder_ret = d2 % divisor;
    return;
  }
  int s = countl_zero(divisor);
  divisor <<= s;
  high = high << s | low >> (63 - s) >> 1;
  low <<= s;
  uint64_t qt1 = (high / ((divisor >> 32) + 1)) << 32;
  uint128 ml1 = int128_internal::Mul64x64(qt1, divisor);
  high -= Uint128High64(ml1) + (low < Uint128Low64(ml1));
  low -= Uint128Low64(ml1);
  uint64_t qt2 = (high << 30 | low >> 34) / ((divisor >> 34) + 1);
  uint128 ml2 = int128_internal::Mul64x64(qt2, divisor);
  high -= Uint128High64(ml2) + (low < Uint128Low64(ml2));
  low -= Uint128Low64(ml2);
  if(ABSL_PREDICT_FALSE(high == 0)) {
    *quotient_ret = qt1 + qt2 + low / divisor;
    *remainder_ret = (low % divisor) >> s;
    return;
  }
  int t = countl_zero(high);
  high = (high << t) | (low >> (64 - t));
  low <<= t;
  uint64_t result = high >= divisor;
  uint64_t current = high - (high >= divisor) * divisor;
  for (int i = 0; i != 64 - t; ++i) {
    uint64_t large = current >> 63;
    current = (current << 1) | (low >> 63);
    low <<= 1;
    large |= (current >= divisor);
    result = (result << 1) | large;
    current -= divisor & (0 - large);
  }
  *quotient_ret = result + qt1 + qt2;
  *remainder_ret = current >> s;
#endif
}

inline void DivModImpl(uint128 dividend, uint128 divisor, uint128* quotient_ret,
                       uint128* remainder_ret) {
  uint64_t dividend_high = Uint128High64(dividend);
  uint64_t dividend_low = Uint128Low64(dividend);
  uint64_t divisor_high = Uint128High64(divisor);
  uint64_t divisor_low = Uint128Low64(divisor);

  assert(divisor_high != 0 || divisor_low != 0);

  if (divisor_high == 0) {
    if (dividend_high >= divisor_low) {
      // Long division/modulo
      uint64_t qt_high = dividend_high / divisor_low;
      uint64_t rem_tmp = dividend_high % divisor_low;
      uint64_t qt_low;
      uint64_t rem_result;
      DivModImpl(MakeUint128(rem_tmp, dividend_low), divisor_low, &qt_low,
                 &rem_result);
      *quotient_ret = MakeUint128(qt_high, qt_low);
      *remainder_ret = MakeUint128(0, rem_result);

    } else {
      // If the quotient fits within 64 bits
      uint64_t qt_result;
      uint64_t rem_result;
      DivModImpl(MakeUint128(dividend_high, dividend_low), divisor_low,
                 &qt_result, &rem_result);
      *quotient_ret = MakeUint128(0, qt_result);
      *remainder_ret = MakeUint128(0, rem_result);
    }
  } else if (dividend_high >= divisor_high) {
    int shift = countl_zero(divisor_high);
    uint64_t xhigh = dividend_high >> (63 - shift) >> 1;
    uint64_t xlow =
        (dividend_high << shift) | (dividend_low >> (63 - shift) >> 1);
    uint64_t yhigh =
        (divisor_high << shift) | (divisor_low >> (63 - shift) >> 1);
    uint64_t ylow = divisor_low << shift;
    uint64_t threshold = (uint64_t(2) << shift) - 1;

    uint64_t qt;
    uint64_t rem;
    DivModImpl(MakeUint128(xhigh, xlow), yhigh, &qt, &rem);
    if (ABSL_PREDICT_FALSE(rem <= threshold)) {
      uint128 tmp = int128_internal::Mul64x64(qt, ylow);
      uint64_t thigh = Uint128High64(tmp);
      uint64_t tlow = Uint128Low64(tmp);
      qt -= (rem < thigh || (rem == thigh && (dividend_low << shift) < tlow));
    }

    *quotient_ret = MakeUint128(0, qt);
    uint128 fl = int128_internal::Mul64x64(qt, divisor_low);
    uint64_t fl_high = Uint128High64(fl);
    uint64_t fl_low = Uint128Low64(fl);
    uint64_t rem_low = dividend_low - fl_low;
    uint64_t rem_high =
        dividend_high - fl_high - divisor_high * qt - (dividend_low < fl_low);
    *remainder_ret = MakeUint128(rem_high, rem_low);
  } else {
    // dividend < divisor
    *quotient_ret = 0;
    *remainder_ret = MakeUint128(dividend_high, dividend_low);
  }
}

template <typename T>
uint128 MakeUint128FromFloat(T v) {
  static_assert(std::is_floating_point<T>::value, "");

  // Rounding behavior is towards zero, same as for built-in types.

  // Undefined behavior if v is NaN or cannot fit into uint128.
  assert(std::isfinite(v) && v > -1 &&
         (std::numeric_limits<T>::max_exponent <= 128 ||
          v < std::ldexp(static_cast<T>(1), 128)));

  if (v >= std::ldexp(static_cast<T>(1), 64)) {
    uint64_t hi = static_cast<uint64_t>(std::ldexp(v, -64));
    uint64_t lo = static_cast<uint64_t>(v - std::ldexp(static_cast<T>(hi), 64));
    return MakeUint128(hi, lo);
  }

  return MakeUint128(0, static_cast<uint64_t>(v));
}

#if defined(__clang__) && (__clang_major__ < 9) && !defined(__SSE3__)
// Workaround for clang bug: https://bugs.llvm.org/show_bug.cgi?id=38289
// Casting from long double to uint64_t is miscompiled and drops bits.
// It is more work, so only use when we need the workaround.
uint128 MakeUint128FromFloat(long double v) {
  // Go 50 bits at a time, that fits in a double
  static_assert(std::numeric_limits<double>::digits >= 50, "");
  static_assert(std::numeric_limits<long double>::digits <= 150, "");
  // Undefined behavior if v is not finite or cannot fit into uint128.
  assert(std::isfinite(v) && v > -1 && v < std::ldexp(1.0L, 128));

  v = std::ldexp(v, -100);
  uint64_t w0 = static_cast<uint64_t>(static_cast<double>(std::trunc(v)));
  v = std::ldexp(v - static_cast<double>(w0), 50);
  uint64_t w1 = static_cast<uint64_t>(static_cast<double>(std::trunc(v)));
  v = std::ldexp(v - static_cast<double>(w1), 50);
  uint64_t w2 = static_cast<uint64_t>(static_cast<double>(std::trunc(v)));
  return (static_cast<uint128>(w0) << 100) | (static_cast<uint128>(w1) << 50) |
         static_cast<uint128>(w2);
}
#endif  // __clang__ && (__clang_major__ < 9) && !__SSE3__
}  // namespace

uint128::uint128(float v) : uint128(MakeUint128FromFloat(v)) {}
uint128::uint128(double v) : uint128(MakeUint128FromFloat(v)) {}
uint128::uint128(long double v) : uint128(MakeUint128FromFloat(v)) {}

uint128 operator/(uint128 lhs, uint128 rhs) {
  uint128 quotient = 0;
  uint128 remainder = 0;
  DivModImpl(lhs, rhs, &quotient, &remainder);
  return quotient;
}

uint128 operator%(uint128 lhs, uint128 rhs) {
  uint128 quotient = 0;
  uint128 remainder = 0;
  DivModImpl(lhs, rhs, &quotient, &remainder);
  return remainder;
}

namespace {

std::string Uint128ToFormattedString(uint128 v, std::ios_base::fmtflags flags) {
  // Select a divisor which is the largest power of the base < 2^64.
  uint128 div;
  int div_base_log;
  switch (flags & std::ios::basefield) {
    case std::ios::hex:
      div = 0x1000000000000000;  // 16^15
      div_base_log = 15;
      break;
    case std::ios::oct:
      div = 01000000000000000000000;  // 8^21
      div_base_log = 21;
      break;
    default:  // std::ios::dec
      div = 10000000000000000000u;  // 10^19
      div_base_log = 19;
      break;
  }

  // Now piece together the uint128 representation from three chunks of the
  // original value, each less than "div" and therefore representable as a
  // uint64_t.
  std::ostringstream os;
  std::ios_base::fmtflags copy_mask =
      std::ios::basefield | std::ios::showbase | std::ios::uppercase;
  os.setf(flags & copy_mask, copy_mask);
  uint128 high = v;
  uint128 low;
  DivModImpl(high, div, &high, &low);
  uint128 mid;
  DivModImpl(high, div, &high, &mid);
  if (Uint128Low64(high) != 0) {
    os << Uint128Low64(high);
    os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
    os << Uint128Low64(mid);
    os << std::setw(div_base_log);
  } else if (Uint128Low64(mid) != 0) {
    os << Uint128Low64(mid);
    os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
  }
  os << Uint128Low64(low);
  return os.str();
}

}  // namespace

std::string uint128::ToString() const {
  return Uint128ToFormattedString(*this, std::ios_base::dec);
}

std::ostream& operator<<(std::ostream& os, uint128 v) {
  std::ios_base::fmtflags flags = os.flags();
  std::string rep = Uint128ToFormattedString(v, flags);

  // Add the requisite padding.
  std::streamsize width = os.width(0);
  if (static_cast<size_t>(width) > rep.size()) {
    const size_t count = static_cast<size_t>(width) - rep.size();
    std::ios::fmtflags adjustfield = flags & std::ios::adjustfield;
    if (adjustfield == std::ios::left) {
      rep.append(count, os.fill());
    } else if (adjustfield == std::ios::internal &&
               (flags & std::ios::showbase) &&
               (flags & std::ios::basefield) == std::ios::hex && v != 0) {
      rep.insert(size_t{2}, count, os.fill());
    } else {
      rep.insert(size_t{0}, count, os.fill());
    }
  }

  return os << rep;
}

namespace {

uint128 UnsignedAbsoluteValue(int128 v) {
  // Cast to uint128 before possibly negating because -Int128Min() is undefined.
  return Int128High64(v) < 0 ? -uint128(v) : uint128(v);
}

}  // namespace

#if !defined(ABSL_HAVE_INTRINSIC_INT128)
namespace {

template <typename T>
int128 MakeInt128FromFloat(T v) {
  // Conversion when v is NaN or cannot fit into int128 would be undefined
  // behavior if using an intrinsic 128-bit integer.
  assert(std::isfinite(v) && (std::numeric_limits<T>::max_exponent <= 127 ||
                              (v >= -std::ldexp(static_cast<T>(1), 127) &&
                               v < std::ldexp(static_cast<T>(1), 127))));

  // We must convert the absolute value and then negate as needed, because
  // floating point types are typically sign-magnitude. Otherwise, the
  // difference between the high and low 64 bits when interpreted as two's
  // complement overwhelms the precision of the mantissa.
  uint128 result = v < 0 ? -MakeUint128FromFloat(-v) : MakeUint128FromFloat(v);
  return MakeInt128(int128_internal::BitCastToSigned(Uint128High64(result)),
                    Uint128Low64(result));
}

}  // namespace

int128::int128(float v) : int128(MakeInt128FromFloat(v)) {}
int128::int128(double v) : int128(MakeInt128FromFloat(v)) {}
int128::int128(long double v) : int128(MakeInt128FromFloat(v)) {}

int128 operator/(int128 lhs, int128 rhs) {
  assert(lhs != Int128Min() || rhs != -1);  // UB on two's complement.

  uint128 quotient = 0;
  uint128 remainder = 0;
  DivModImpl(UnsignedAbsoluteValue(lhs), UnsignedAbsoluteValue(rhs),
             &quotient, &remainder);
  if ((Int128High64(lhs) < 0) != (Int128High64(rhs) < 0)) quotient = -quotient;
  return MakeInt128(int128_internal::BitCastToSigned(Uint128High64(quotient)),
                    Uint128Low64(quotient));
}

int128 operator%(int128 lhs, int128 rhs) {
  assert(lhs != Int128Min() || rhs != -1);  // UB on two's complement.

  uint128 quotient = 0;
  uint128 remainder = 0;
  DivModImpl(UnsignedAbsoluteValue(lhs), UnsignedAbsoluteValue(rhs),
             &quotient, &remainder);
  if (Int128High64(lhs) < 0) remainder = -remainder;
  return MakeInt128(int128_internal::BitCastToSigned(Uint128High64(remainder)),
                    Uint128Low64(remainder));
}
#endif  // ABSL_HAVE_INTRINSIC_INT128

std::string int128::ToString() const {
  std::string rep;
  if (Int128High64(*this) < 0) rep = "-";
  rep.append(Uint128ToFormattedString(UnsignedAbsoluteValue(*this),
                                      std::ios_base::dec));
  return rep;
}

std::ostream& operator<<(std::ostream& os, int128 v) {
  std::ios_base::fmtflags flags = os.flags();
  std::string rep;

  // Add the sign if needed.
  bool print_as_decimal =
      (flags & std::ios::basefield) == std::ios::dec ||
      (flags & std::ios::basefield) == std::ios_base::fmtflags();
  if (print_as_decimal) {
    if (Int128High64(v) < 0) {
      rep = "-";
    } else if (flags & std::ios::showpos) {
      rep = "+";
    }
  }

  rep.append(Uint128ToFormattedString(
      print_as_decimal ? UnsignedAbsoluteValue(v) : uint128(v), os.flags()));

  // Add the requisite padding.
  std::streamsize width = os.width(0);
  if (static_cast<size_t>(width) > rep.size()) {
    const size_t count = static_cast<size_t>(width) - rep.size();
    switch (flags & std::ios::adjustfield) {
      case std::ios::left:
        rep.append(count, os.fill());
        break;
      case std::ios::internal:
        if (print_as_decimal && (rep[0] == '+' || rep[0] == '-')) {
          rep.insert(size_t{1}, count, os.fill());
        } else if ((flags & std::ios::basefield) == std::ios::hex &&
                   (flags & std::ios::showbase) && v != 0) {
          rep.insert(size_t{2}, count, os.fill());
        } else {
          rep.insert(size_t{0}, count, os.fill());
        }
        break;
      default:  // std::ios::right
        rep.insert(size_t{0}, count, os.fill());
        break;
    }
  }

  return os << rep;
}

ABSL_NAMESPACE_END
}  // namespace absl
