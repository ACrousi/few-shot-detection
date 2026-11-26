# 建置與執行

完整計算流程由平台自行實作，見 [平台架構](../../docs/platform-architecture.md)。
啟用 `platform_config` 時，模型產物與 SAM3 執行參數以該檔為準。

## 啟動

從工作區根目錄執行，相機與辨識服務使用同一位使用者（容器預設 `admin`）：

```bash
bash applications/few_shot_detection/run_camera.sh
# 另一個終端
bash applications/few_shot_detection/run.sh
```

開啟 <http://127.0.0.1:8090>。工作區根目錄的 `run.sh` 轉送至應用入口；host 可使用根目錄 `start.sh` 啟動開發容器。

`run.sh` 載入 ROS Jazzy、專案 `.venv`、TensorRT 動態函式庫與 Fast DDS 設定，再由 `launch_pipeline.py` 載入 [dinov3_cases.yaml](config/dinov3_cases.yaml)。可直接追加 ROS 參數：

```bash
bash applications/few_shot_detection/run.sh --ros-args -p port:=8091
```

相機使用已安裝的 `orbbec_camera`；若位於其他 ROS 工作區，可設定 `ORBBEC_SETUP=/path/to/install/setup.bash`。RGB、深度與 CameraInfo 訂閱採 RELIABLE、KEEP_LAST、depth=1。跨使用者啟動可能影響 Fast DDS 共享記憶體通訊。

相機入口使用 `enumerate_net_device:=false`，停用區網相機列舉。相機與辨識入口均強制
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`，並清除 `ROS_STATIC_PEERS`、`ROS_DISCOVERY_SERVER`
與舊的 `ROS_LOCALHOST_ONLY` 設定。預設 Fast DDS UDP 僅允許 `127.0.0.1`，保留共享記憶體傳輸。
若自行指定 DDS profile，請同樣限制網路介面。修改後須重啟相機與辨識程序才會生效。

## 環境與模型

需由系統提供 Python 3.12、ROS 2 Jazzy、CUDA driver/runtime、cv_bridge、vision_msgs、isaac_ros_apriltag_interfaces 與相機驅動。Python 相依套件與模型均位於 `applications/few_shot_detection`，不需要相鄰模型專案。

```bash
# 在 Isaac ROS 容器內，以工作區使用者執行；不要對整支入口加 sudo
bash applications/few_shot_detection/setup_all.sh
```

`setup_all.sh` 先呼叫 `setup_env.sh`，再檢查 OpenCV KCF；不再建置自訂 CUDA 函式庫。入口不下載或建置模型。

個別維護可執行 `bash applications/few_shot_detection/setup_env.sh`。GPU ROI 使用預編譯 PyTorch CUDA wheel，不需另外編譯。

`setup_env.sh` 先安裝 CUDA 13.0 Torch／torchvision，再使用 [constraints-thor.txt](constraints-thor.txt) 安裝 [requirements.txt](requirements.txt)。可透過 `PYTHON_BIN`、`DINOV3_VENV` 指定 Python 與環境位置。

| 用途 | 模型與執行方式 | 本機位置 |
| --- | --- | --- |
| SAM3 vision | 官方 Torch-TensorRT FP16，1008×1008 | `artifacts/sam3/vision-fp16.ts` |
| SAM3 文字與 bbox head | 官方 Transformers PyTorch FP16 | `artifacts/sam3/source/` |
| Cat2Real | 官方 Torch-TensorRT FP32，384×384 | `artifacts/cat2real-dinov3-384/cat2real-fp32-trt.ts` |
| Cat2Real processor／原始權重 | Transformers | `artifacts/cat2real-dinov3-384/source/` |
| 商品圖庫 | 依資料夾名稱分類 | `data/gallery/` |

SAM3 子程序與主程序共用 `.venv`，透過共享影格池傳遞 RGB。KCF 在主程序的 CPU 執行。

模型建置統一使用 `build_models.py`：

```bash
# 已有本機權重時重建全部模型
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py

# 分別建置
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --target sam3
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --target cat2real

# 首次下載；SAM3 需先取得官方存取權並登入
applications/few_shot_detection/.venv/bin/hf auth login
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --download

# 只驗證既有產物
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --validate-only
```

Cat2Real 固定 revision `a763964f2041722ffcc5bc4c72e3cb241530727e`，SAM3 固定 `3c879f39826c281e95690f02c7821c4de09afae7`。建置要求完整編譯 vision 圖；儲存後重新載入，與原始 FP32 權重比較，通過才寫入同名 `.json` 部署中繼資料。各模型目錄的 `validation.json` 保存 SHA256、版本、樣本結果與測速。

Cat2Real 在 Thor 固定使用 FP32，避免 FP16 非有限輸出。SAM3 只編譯 vision；bbox head 與文字編碼使用官方 PyTorch 模組，不執行 mask 或 SAM3 tracking。

## GPU 延遲優化

SAM3 使用 GPU 前處理，DINO 在同一影格內批次處理 ROI，再以矩陣比對圖庫。3D 框使用 CPU 深度連通區與 Open3D CPU RANSAC 平面擬合。`bbox3d_quantile` 目前部署值為 `0.07`。

GPU ROI 透過 PyTorch CUDA 切片、置中白色補邊與雙線性縮放，輸出 uint8 NHWC，再交給既有批次正規化與模型。無需 `nvcc`、CMake 或自訂 `.so`；平面擬合沿用 Open3D CPU。

Compose 使用 `deployment/docker/Dockerfile.isaac-ros`，沿用 Isaac ROS base image，不再安裝或編譯 CUDA geometry。

另建並驗證模型，不覆蓋原有單筆引擎：

```bash
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --target cat2real --max-batch-size 8
applications/few_shot_detection/.venv/bin/python applications/few_shot_detection/build_models.py --target sam3 --artifact-suffix native
```

驗證通過後寫入同名 `.json`，啟動時優先選用 `cat2real-fp32-batch8-trt.ts` 與 `vision-fp16-native.ts`。DINO 保持 FP32，支援 batch 1–8，僅合併本次影格已有的物件，不等待後續影格。沒有新引擎時沿用原有單筆版本。TensorRT 引擎應在實際部署 GPU 上重新建置。

疊圖統一走 CPU，發布 ROS Image 後由網頁 bridge 做背景 JPEG 轉換。GPU 前處理與 OpenCV/PIL 的插值可能有一個 RGB 灰階的差異，描述子與 SAM3 偵測框的數值驗證流程維持原樣。


## 設定與排程

設定以 [dinov3_cases.yaml](config/dinov3_cases.yaml) 為準；以下是目前檔案值，而非節點內建預設值。

| 參數 | 目前值 | 用途 |
| --- | --- | --- |
| `grounding_prompt` | `box, can.` | 1–4 個逗號分隔類別，文字特徵按 prompt 快取 |
| `grounding_detection_interval` | 0.1 | 背景 SAM3＋DINO 工作啟動的最小間隔（秒）；0.0 在空閒且有新影格時提交，1.0 為每秒一次；KCF 始終啟用 |
| `grounding_box_threshold` | 0.6 | SAM3 偵測門檻 |
| `grounding_containment_threshold` | 0.5 | 追蹤後去除被包含的小框；0 停用 |
| `recognition_threshold` | 0.6 | DINO cosine 候選門檻，再以 Lab 色彩相似度選分類 |
| `tracker_width`／`tracker_height` | 640／384 | 追蹤影像最大寬高，維持原比例 |
| `kcf_detect_threshold` | 0.5 | OpenCV KCF response 門檻，非分類機率 |
| `tracker_max_missed_seconds` | 0.5 | 自最後可見輸出起算的失追寬限秒數，期間隱藏舊框；0.0 停用 |
| `bbox_smoothing_seconds` | 0.0 | 框移動平均時間視窗（秒）；0.0 停用 |
| `pose_sync_tolerance` | 6 | RGB／深度同步容許影格數，以 `video_fps` 換算秒數；0 要求完全相同 |
| `bbox3d_max_age` | 30 | 3D 快取有效影格數，以 `video_fps` 換算秒數，須大於 0 |
| `bbox_visualization_mode` | `3d` | 疊圖模式，亦可選 `2d` |

追蹤排程、失追寬限與框平滑以 monotonic 秒數計算。深度同步及 3D 快取期限的影格設定見下方說明。

包含框過濾在每次追蹤更新及新舊 context 合併後，使用目前框座標計算「交集面積 ÷ 較小框面積」；達到門檻便移除小框及其追蹤身分。IoU NMS 的參數與邏輯已移除。設定修改後需重啟服務。

背景 3D 量測對每張新處理 RGB 影格提交工作，與 SAM3 偵測間隔無關；忙碌時只保留一份最新待算輸入，相同影格重跑不重複提交。

KCF 參數位於 `config/dinov3_cases.yaml`。追蹤縮放不改變模型裁切及深度處理的原圖座標。

非同步流程先交接 SAM3 框，再更新 DINO 分類，期間前景 KCF 持續追蹤。交接只更新仍存活的 seed owner，不回退目前框。SAM3 漏偵測不再淘汰追蹤，原生追蹤有效的物件持續保留原 context；失追物件在寬限內隱藏、重試偵測並保留公開 ID。配對是幾何關聯，沒有長期 ReID。

`recognition_margin` 為舊 API 相容欄位，已不影響 Lab 分類，因此未列入 YAML。`score` 是所選參考圖的 DINO 分數；`margin` 回傳值是候選類別間的 Lab 分數差。

## 驗證與程式導覽

[測試說明](tests/README.md) 列出目前回歸與 GPU 整合檢查。[程式導覽](src/dinov3_cases/README.md) 說明模組責任、影格租約、3D 深度框、ROS topics 與 ROS Image／MJPEG 介面。

API `result_fps` 只計不同 RGB 影格的完成率；背景結果刷新不增加此計數。SAM3、KCF、3D、發布與 JPEG 轉換耗時應分別判讀，模型單獨測速不代表完整流程 FPS。

`pose_sync_tolerance` 使用非負整數影格數，以 `video_fps` 換算 RGB／深度時間戳容許差距（秒 = 影格數 / FPS），不是相機實測 FPS 或佇列索引。部署設定為 6 幀，在 30 FPS 下等同原本 0.2 秒；程式預設為 2 幀。0 表示時間戳必須相同，不等待未來深度。舊的秒數設定與命令列覆寫需改為整數幀數。

`bbox3d_max_age` 同樣以 `video_fps` 將正整數幀數換算為有效秒數；影像停止時仍按經過時間淘汰過期快取。原本部署的 1.0 秒改為 30 幀（30 FPS），程式預設 1.5 秒改為 45 幀。舊的秒數設定與命令列覆寫需改為整數幀數；`bbox3d_age_seconds` 輸出仍以秒表示。
