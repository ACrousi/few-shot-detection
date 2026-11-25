# 平台自行擁有完整 pipeline

`dinov3_cases/platforms/nvidia/pipeline.py` 是完整 NVIDIA pipeline，沒有共用 stage
factory、provider、Protocol 或 tensor backend。其他平台可自行安排資料流、記憶體
與模型執行，不必依照 NVIDIA 的內部步驟。

```text
ROS node / 單張 CLI
    → 選擇 platforms/nvidia/pipeline.py
        → 模型、圖庫、前處理、傳輸、追蹤交接、3D 與清理
```

| 檔案 | 責任 |
|---|---|
| `platforms/__init__.py` | 簡單選擇已實作平台；不建立 stage 或註冊 backend |
| `platforms/nvidia/pipeline.py` | 平台設定、模型載入、單張辨識、資源生命週期 |
| `platforms/nvidia/live.py` | 完整即時排程、SAM3/KCF 交接、DINO 分類與結果整合 |
| `platforms/nvidia/recognition.py` | GPU 圖庫特徵、DINO／Lab 比對 |
| `platforms/nvidia/depth.py` | 深度轉換、幾何資料路徑與 3D 結果 |
| `platforms/nvidia/transport.py` | 平台自選的 SAM3 子程序／共享記憶體 |
| `platforms/nvidia/encoder.py`、`sam3.py`、`trt_backend.py` | 原生模型執行 |
| `platforms/nvidia/geometry.py`、`cpu_roi.py` | NVIDIA pipeline 內選用的 CUDA／CPU 工具 |
| `platforms/nvidia/artifacts.py` | NVIDIA 引擎、權重及工具預設路徑 |
| `runtime.py` | 平台中立的 application 目錄、圖庫及模型規格 |

舊 `inference/encoder.py`、`sam3_trt.py`、`trt_backend.py`、`grounding_ipc.py`、
`gpu_geometry.py`、`platforms/factory.py`、`contracts.py` 已移除；直接 import 真正的實作。
圖庫檔案管理、輸入參數驗證、KCF 與通用幾何演算法仍可作為工具重用。

## 設定

ROS 的 `platform_config` 預設為 `config/platforms/nvidia.yaml`，相對於
`FEW_SHOT_HOME`（預設 application 目錄）。單張 CLI 的 `platform_config` 則相對於
`config/native.yaml`。平台檔的模型路徑相對於平台 YAML。

```yaml
platform: nvidia
pipeline: dinov3_cases.platforms.nvidia.pipeline
models:
  sam3:
    artifact: ../../artifacts/sam3/vision-fp16-native.ts
  cat2real:
    artifact: ../../artifacts/cat2real-dinov3-384/cat2real-fp32-batch8-trt.ts
operations:
  roi: cuda
execution:
  sam3: subprocess
  timeout: 60.0
```

`models`、`operations` 與 `execution` 都是 NVIDIA pipeline 自己的設定，suite 不驗證
這些欄位的統一 schema。Intel／Qualcomm 可以使用完全不同的欄位與內部實作。
`roi: cpu` 使用 CPU 裁切及正規化。深度框由 Open3D CPU RANSAC 擬合主平面，再計算旋轉框與投影校正。

已移除舊 ROS `engine_path`、`grounding_engine_path`、`grounding_worker_python`、
`grounding_ipc_timeout`、`bbox3d_plane_backend` 及 CLI `runtime` 相容入口；請改平台 YAML。
CPU／CUDA ROI 選項不代表模型 CPU fallback。Intel／Qualcomm pipeline 尚未實作。

## 啟動與驗證

```bash
# 一般 ROS 與單張 CLI 入口
./run.sh
./infer.sh --image applications/few_shot_detection/artifacts/validation/person.jpg --prompt person

# 由 suite 直接委派平台程式檢查（使用應用 venv）
applications/few_shot_detection/.venv/bin/python -m rsctl validate \
  --config applications/few_shot_detection/config/platforms/nvidia.yaml
```

加 `--require-runtime` 可實際載入模型；執行時使用 `infer.sh` 相同的 TensorRT/CUDA
library 環境。完整回歸用 `./verify.sh`；實機模型與非同步整合用 `./verify.sh --hardware`。

## 資料搬運

suite 不傳遞任何中間 tensor。NVIDIA 內部保留既有的 CPU RGB、KCF、SAM3 共享記憶體
子程序與 DINO GPU 路徑。此次調整使平台可自由改進這些步驟，並未宣稱已消除 SAM3／DINO
之間重複上傳或完成 GPU zero-copy。共用 HTTP／ROS 接收最終結果與診斷物件。

## 2026-11-17 驗證

Suite 31 項、應用 564 項回歸通過；最後參考圖路徑調整另有 26 項針對性測試通過。
實機 SAM3、DINO ROI 比對、CUDA 平面擬合、KCF 非同步、ROS/HTTP 與單張 CLI 均通過。
CPU/CUDA ROI 單樣本最大像素差 1，DINO cosine 0.9999284。
SDK、CLI 與應用 wheel 可離線建置，封裝未含已刪除的 backend／轉接模組。
證據在 `validation/platform-owned-pipeline/`；這不是 Intel／Qualcomm 或 zero-copy 驗證。
