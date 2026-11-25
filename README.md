# few-shot-detection

由 `suite-inference` 複製，依其 application / model adapter / native runtime 邊界移入
`isaac_ros-dev/dinov3_ws` 的 SAM3 + Cat2Real few-shot 辨識、CPU KCF 追蹤、RGB-D 3D 框、ROS 2 與 HTTP/MJPEG。
共用 SDK 與 rsctl 不包含本應用的模型或任務規則。正式推論使用應用既有 Torch-TensorRT runtime；沒有 mock 或自動 CPU fallback。

## 目前這台機器直接執行

```bash
cd /workspaces/few-shot-detection
# 終端機 1：RGB-D 相機
./run_camera.sh
# 終端機 2：辨識、追蹤、3D 與 Web
./run.sh
```

開啟 http://127.0.0.1:8090 。首頁可切換相機／上傳影像，`/images` 管理各類別參考圖。
網頁的即時調整不會寫回設定檔。按 Ctrl-C 停止各程序。
既有 ROS node/package 名稱與 `/medicine_recognition/*` topics 保留，以相容原使用端。
若與其他實例並行，請給相機及推論相同且獨立的 `ROS_DOMAIN_ID`，並以 `./run.sh -p port:=8091` 選擇不同 HTTP port。

不需要 ROS 的單張真實模型推論：

```bash
./infer.sh --image applications/few_shot_detection/artifacts/validation/person.jpg \
  --prompt person --output validation/standalone.json
```

`person.jpg` 用於偵測驗證；現有圖庫為藥品，所以人物應被拒識為 `unknown`。
換成自己的圖庫時，每類放一個目錄：`data/gallery/<類別>/*.png`，或設定 `config/native.yaml` 的 `gallery`。
輸出含原圖 pixel XYXY、SAM3 detection score、類別、DINO cosine、Lab 相似度、拒識原因與參考影像。
相似度不是校準機率。

## KCF 追蹤

追蹤使用 CPU OpenCV KCF。
SAM3/Cat2Real 與 CUDA 幾何仍使用原本 NVIDIA 路徑；這次變更不代表完整 Intel／Qualcomm 移植完成。
追蹤設定與驗證見 [KCF 說明](docs/kcf-tracking.md)。

## 專案結構與設定

- `packages/robotic_inference_sdk`、`tools/rsctl`：原 Suite 共用 SDK／CLI；維持通用。
- `applications/few_shot_detection/src/few_shot_detection`：無 ROS pipeline 與 CLI。
- `applications/few_shot_detection/src/dinov3_cases/dinov3_cases`：保留既有模型 adapters (`inference`)、辨識、追蹤、幾何、ROS 與 API；不依賴來源 checkout。
- `applications/few_shot_detection/config/native.yaml`：無 ROS 應用設定；不是 `rsctl validate` 的 tensor schema。
- `applications/few_shot_detection/config/dinov3_cases.yaml`：相機／ROS／3D／偵測與追蹤部署設定。
- `applications/few_shot_detection/artifacts`、`data/gallery`、`.venv`：此機器已複製的獨立資產／資料／環境；不納入 Git。
- `drivers/orbbec`：相機驅動來源與此機器已安裝的 runtime。
- `compose.yaml`、`start.sh`：獨立的 `few-shot-detection` Compose 專案，供 Jetson host 啟動開發容器。

模型 native GPU tensor 路徑留在應用層，不強迫經過 SDK 的 CPU NumPy 介面。
ROS 節點負責非同步排程、topic 與發布；無 ROS CLI 重用相同的 SAM3/Cat2Real adapters 及 gallery matching。
可設定 `FEW_SHOT_HOME` 指定已安裝應用的設定／模型／資料根目錄；預設為本 checkout 的 application 目錄。

## 新機器建置

本機複製的環境和 TensorRT engine 僅驗證於 NVIDIA Thor / ARM64；不同 GPU 必須重新建置模型。

```bash
# Jetson host：建立／進入容器
./start.sh
# 容器內
./setup.sh
bash drivers/orbbec/build.sh
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --download
```

SAM3 下載需已取得 Hugging Face 存取權。詳細版本／模型建置請見
[BUILD_GUIDE](applications/few_shot_detection/BUILD_GUIDE.md)。相機可用 `ORBBEC_SETUP=/path/to/install/setup.bash` 指定其他安裝。
新機器需自行準備圖庫。pip 安裝入口：`python -m pip install -e packages/robotic_inference_sdk -e tools/rsctl -e applications/few_shot_detection`。

## 驗證

```bash
./verify.sh             # SDK + application regression（需已安裝測試依賴及 ROS）
./verify.sh --hardware  # 再跑真實模型、IPC、HTTP/ROS、非同步追蹤
```

驗證證據與限制見 [本次移入驗證](docs/few-shot-validation.md)。Suite 原始說明見 [suite-readme](docs/suite-readme.md)。

## 平台執行流程

每個平台自行實作完整 pipeline。NVIDIA 入口為 `platforms/nvidia/pipeline.py`，
包含模型、圖庫、前處理、傳輸與即時運算；suite 只負責啟動及環境管理。
`config/platforms/nvidia.yaml` 選擇平台程式與設定。Intel／Qualcomm 尚未實作。
舊 stage 工廠、共用 backend 及相容轉接已移除，詳見 [平台架構](docs/platform-architecture.md)。
