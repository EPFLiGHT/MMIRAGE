"""Batch API processor implementation.

This module provides a processor that serializes inference requests and submits
them to a provider batch API (e.g. OpenAI Batch) instead of running a model
locally. Results are retrieved asynchronously by the receiver utilities.
"""
