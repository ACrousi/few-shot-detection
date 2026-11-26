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
| `platforms/nvidia/live_inference.py` | 完整即時排程、SAM3/KCF 交接、DINO 分類與結果整合 |
| `platforms/nvidia/gallery_matching.py` | GPU 圖庫特徵、DINO／Lab 比對 |
| `ros/depth.py` | 深度轉換、幾何資料路徑與 3D 結果 |
| `platforms/nvidia/sam3_worker.py` | 平台自選的 SAM3 子程序／共享記憶體 |
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

## Python 套件根名遷移評估

建議暫緩將 `dinov3_cases` 統一為 `few_shot_detection`。目前使用者已有產品名稱的 CLI。若日後要對外提供 Python SDK，或要統一管理 wheel 與 ROS 安裝的程式碼，可以重新評估。

| 範圍 | 現有使用者與影響 | 遷移時的處理 |
|---|---|---|
| Python import（內部實作） | CLI、模型、Web、ROS、建模與診斷工具及測試都會載入舊根名。mock 的字串路徑也參與執行。 | 一次遷移內部 import 與 patch 目標。同一程序中的類別、例外與模組狀態應只有一份。 |
| pip（安裝介面） | 應用 distribution 名為 `few-shot-detection`。應用 setup.py 同時封裝兩個 Python 根，並把 HTML／JS／CSS 歸在 `dinov3_cases.api`。`runtime_env.sh` 可直接從兩個來源路徑載入，因此 checkout 測試可能掩蓋 wheel 漏包。 | 保留 distribution 名、`few-shot-detection` 命令與 `python -m few_shot_detection`。調整 package discovery、package data、來源路徑與 runtime.py 的路徑推導。從 checkout 外分別測 editable、乾淨 wheel 與 `FEW_SHOT_HOME`。 |
| ROS（公開部署介面） | package.xml、ament resource marker、share 目錄、setup.cfg 的 `lib/dinov3_cases` 和 ROS console script 共同提供 `ros2 run dinov3_cases case_recognition`。ROS node 名稱也是參數 YAML 的首層 key。 | 保留 ROS package、node 名稱、參數檔、topics 與執行檔名稱，只改 console script 的 Python callable。以獨立安裝前綴驗證 colcon/ament 能載入相同實作及網頁資源，避免 pip 與 ROS 各覆蓋同名套件的一部分。 |
| 平台 YAML（部署設定與輸出） | NVIDIA YAML 指定完整模組名，NVIDIA 建構子核對該字串，Suite 的 `rsctl` 依 YAML 值啟動模組。單張 CLI 也將該值輸出到 JSON 的 `runtime`。Intel／Qualcomm 檔案仍是樣板。 | 舊 NVIDIA 模組保留可用 `-m` 啟動的薄入口，讓新實作接受新舊設定值。以兩份 YAML 跑 `rsctl validate`、`pipeline-run` 及單張推論，核對 JSON 契約與未實作平台的錯誤。 |
| 子程序（跨程序入口） | `launch_pipeline.py` 以 `-m dinov3_cases.ros.node` 啟動節點。SAM3 子程序依 `worker_module` 以選定的 Python 啟動 `dinov3_cases.platforms.nvidia.sam3_worker --fd …`。 | 讓 ROS 舊模組的 `-m` 入口可轉呼叫新實作。父子程序同版部署時同步更新 worker 預設值。若另行供應 worker venv，過渡期在該環境提供舊路徑薄入口並測試真實 socket/共享記憶體握手。 |
| 專案內的外部呼叫點 | `infer.sh`、`run.sh`、`launch_pipeline.py`、`setup_all.sh`、`setup_env.sh`、`build_models.py`、驗證及 profiling 腳本都依賴命令、模組或來源目錄。Suite SDK 以設定指定的模組啟動。此盤點只涵蓋本 checkout。 | 保留 shell 使用方式，調整固定目錄與建模 import。從乾淨安裝執行模型載入、圖庫比對、ROS/HTTP 和 worker 關閉驗證；若有倉庫外呼叫者，再依實際使用的 Python API 增補相容入口。 |

遷移的驗收應涵蓋 `ros2 pkg prefix dinov3_cases`、`ros2 pkg executables dinov3_cases`、原有 YAML 的參數載入、ROS/HTTP 輸出、CLI 實圖辨識、IPC 逾時與共享記憶體回收。除了 `./verify.sh --hardware`，也要在未注入 checkout 的環境驗證安裝後的入口。
