# Platform support

Suite has deployment recipes for NVIDIA, Intel and Qualcomm, and an executable CPU
MOCK template. Recipes do not execute models. Application platform modules own real
runtime support. Run `rsctl validate --config PATH --require-runtime` using the target
application environment, followed by that application's real inference tests.

No automatic CPU or mock fallback is provided. The few-shot application currently
implements NVIDIA only. Intel/Qualcomm require their own model conversion and pipeline.
