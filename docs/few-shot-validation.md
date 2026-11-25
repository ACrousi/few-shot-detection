# few-shot-detection 移入與驗證（2026-11-16）

> 本文件保留移入當時的歷史驗證。追蹤目前使用 KCF，舊追蹤程式與設定已移除；目前建置及驗證見 [KCF 說明](kcf-tracking.md)。

在 NVIDIA Thor / ARM64 / ROS Jazzy 實機執行。來源為兩個工作目錄的當前內容，
包含尚未提交的檔案；未修改來源專案，也未複製其 Git 歷史。
Suite HEAD：`797094667a23b394578025231b90e300e67a5fab`；
isaac_ros-dev HEAD：`6a900459b9a97b650aa8434b4a03166131a47097`。

## 移入內容

Suite SDK、CLI、通用模板與文件保留；完整任務位於 `applications/few_shot_detection`。
SAM3 / Cat2Real Torch-TensorRT、NvDCF、CUDA 3D geometry、gallery matching、HTTP/MJPEG 與 ROS 都已移入。
新增可安裝的 `few-shot-detection` application 與無 ROS pipeline/CLI，設定使用明確的
`native_torch_tensorrt`；共用 SDK 只供配置讀取及未來共用 tensor runtime，不塞入任務規則。
保留原始 Python/ROS package 名稱及 topic 相容性。

約 46 GB 的模型、圖庫、Python 環境與相機安裝檔已獨立複製到本目錄，沒有共用可寫圖庫。
相機驅動來源亦已保留；已安裝相機 driver 的 symlink 轉為實體檔案。
原安裝有一個失效的 CMake export symlink，沒有移入該失效檔；runtime 已驗證，新機器應從 `drivers/orbbec/build.sh` 重建。
模型、venv、driver install 與驗證資料皆由 `.gitignore` 排除。

## 通過項目

| 驗證 | 結果／本機證據 |
|---|---|
| 共用 SDK / rsctl | 56 passed；3 個明確 hardware marker 不在此輪核心測試中執行 |
| 既有應用回歸 | 532 passed，含實機 CUDA/NvDCF regression |
| 新入口、設定與無 ROS 邊界 | 9 passed；合計 application 541 個測試 |
| Python dependency consistency | `python -m pip check` 通過 |
| Native build | CUDA geometry 與 NvDCF helper 從新路徑建置成功；`validation/native-build.log` |
| 真實模型與 IPC | SAM3 bbox IoU 0.9903，16 個 gallery references，1024 維 descriptor，多 prompt 與 IPC parity 通過；`validation/models.json` |
| 無 ROS CLI | `infer.sh` 真實辨識成功；`validation/standalone.json` |
| HTTP / ROS | 上傳、結果先於繪圖、ROS bbox 與 960×540 BGR overlay 發布通過；`validation/ros-http.json` |
| 非同步壓力 | 阻塞 SAM3 / DINO 時追蹤不中斷，geometry 先於 classification、ID 交接及 visualization 通過；`validation/async-nvdcf.json` |
| 實體相機 | RGB 1920×1080、registered depth，約 30 FPS 輸入，7 個追蹤物件及 3D 框；`validation/live-status.json`、`validation/live-detections.json` |
| Web live smoke | `/`、`/images`、`/api/detections` 都為 HTTP 200，MJPEG 收到 JPEG；`validation/live-http.json` |
| 程序獨立性 | 相機與推論程序 mappings 無來源 isaac_ros-dev 路徑；`validation/process-independence.json` |

模型數值是本機固定影像 smoke，不代表任意新圖庫的辨識精度。人物測試圖在藥品圖庫中被正確拒識為 unknown。
相機 FPS 是輸入速率；SAM3 / DINO 非同步執行，不代表每個模型均達 30 FPS。

## 同步修正

- NvDCF 錯誤輸出以暫存檔接收，保留最後 8 KiB 診斷，避免 stderr pipe 塞滿；失敗及 close 會釋放程序、共享記憶體與檔案。
- 修正來源 test_nvdcf_bridge 的過時 import。
- 非同步壓力 fixture 改用人物 ROI reference；等待分類時持續輸入 RGB，接受已完成分類的 dino/cached 狀態，並明確觸發下一個偵測工作，避免依賴 GPU 初始化時間及瞬時狀態。
- venv scripts、相機安裝設定、Compose workspace 與啟動路徑指向新專案。

## 驗證限制

- 容器內沒有 Node.js，因此兩個既有 `.cjs` 前端單元測試未執行；HTTP/MJPEG、前端路由及 Python Web 回歸已驗證。
- 容器內沒有 Docker Compose plugin；Compose YAML 與 build paths 已靜態檢查，尚未在 host 執行新容器 build/up。當前容器內的完整流程已實跑。
- SAM3 權重與原環境直接複製，未重新下載／重新編譯全部 TensorRT 模型。換 GPU／runtime 版本須使用 build_models.py 重建。

## 重現

從 repository root 執行 `./verify.sh`；`./verify.sh --hardware` 追加真實模型、HTTP/ROS 與非同步壓力驗證。
測試隔離 ROS domain 131；請避免另有應用使用該 domain。
首次測試安裝 `applications/few_shot_detection/requirements-dev.txt` 及 SDK 的 `[test,ort]` extras。
日常執行使用 `./run_camera.sh` 與 `./run.sh`，Web 位址 http://127.0.0.1:8090。
