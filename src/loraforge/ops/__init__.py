# SPDX-License-Identifier: Apache-2.0
"""GPU tensor operators, loaded without initializing a device."""

from .flashtsqr import ExecutionInfo, QRFactorization, qr_factor

__all__ = ["ExecutionInfo", "QRFactorization", "qr_factor"]
