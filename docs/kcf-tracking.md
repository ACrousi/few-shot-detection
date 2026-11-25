# CPU KCF tracking

追蹤使用 CPU OpenCV KCF。SAM3/Cat2Real、ROS topics、HTTP 路由及圖庫辨識流程保留。

## 責任與執行

- `tracking/kcf.py`：只依賴 OpenCV-contrib 與 NumPy。每個物件有自己的 KCF instance；RGB 輸入轉 BGR，依最大寬高等比例縮小，輸出映射回原圖 XYXY。
- `tracking/async_tracker.py`：追蹤 context 生命週期、偵測 generation 交接、ID 配對、失追寬限與延後分類。底層 factory 可注入其他追蹤器。
- 底層 context 契約為 `update(pixels, frame_key, detections=None)`、`proposals`、`metadata`、`frame_key`、`close()`。`tracker_id` 只在 context 內有效；`track_id` 是對外穩定 ID。
- SAM3 在背景產生偵測，KCF 將其追到最新影格，再與現有物件配對；分類結果只更新相同 seed owner，不能把框倒退到舊影格。
- 追蹤失敗時立即隱藏框，保留 metadata 至寬限結束；重新偵測可用幾何配對沿用 ID。沒有長期 ReID 保證。
- OpenCV 回傳追蹤成功／失敗，沒有公開可比較的 confidence score；KCF 輸出不偽造 `tracker_confidence`。
- 來源解析度變更但無新偵測時，丟棄舊 tracking state，等待重新偵測。

## 追蹤設定

在 `applications/few_shot_detection/config/dinov3_cases.yaml`：

| 設定 | 預設部署值 | 語意 |
|---|---|---|
| `tracker_width` | 640 | 追蹤影像最大寬度，維持原比例 |
| `tracker_height` | 384 | 追蹤影像最大高度 |
| `tracker_max_missed_seconds` | 0.5 | 失追後隱藏 owner 的保留時間 |
| `kcf_detect_threshold` | 0.5 | OpenCV KCF response threshold，非分類機率 |

SAM3 定期重新偵測負責更新物件尺寸、補回失追；
大幅位移、尺寸快速變化及遮擋場景應另行驗證。CPU 成本隨物件數量增加。
參考 [OpenCV KCF API](https://docs.opencv.org/4.x/d2/dff/classcv_1_1TrackerKCF.html)。

## 建置與平台邊界

`opencv-contrib-python` 已在應用依賴中。單獨移植追蹤時，只需 NumPy 與提供 `TrackerKCF_create` 的 OpenCV；不需要 Torch、TensorRT、CUDA 或 ROS。

`setup_all.sh` 檢查 KCF，並為目前 NVIDIA 模型部署建置 CUDA geometry。
`native/build.sh` 預設只建置 `gpu_geometry`；新 Compose image 名稱為 `few-shot-detection-kcf:local`。
這次替換只移除了追蹤的 NVIDIA 依賴；模型推論與部分 ROI/3D 幾何仍需後續移植。

## 驗證命令

```bash
./verify.sh
# 純 CPU 追蹤及排程測試，可不 source ROS：
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 applications/few_shot_detection/.venv/bin/python -m pytest -q \
  applications/few_shot_detection/tests/test_kcf.py \
  applications/few_shot_detection/tests/test_async_tracker.py
# 含真實模型、ROS/HTTP 與 KCF 非同步測試：
./verify.sh --hardware
```

測試使用臨時 HTTP port／測試 ROS domain，結束會清理程序；不啟動實體相機。

## 本次結果（2026-11-16）

- SDK 56 passed；application 553 passed（含新增的 12 個 KCF 測試）。
- 真實 SAM3/Cat2Real + KCF 非同步阻塞測試、ROS/HTTP smoke 均通過。
- 預設 native build 只建置 CUDA geometry。
- 記錄位於 `validation/kcf/`。測試完成後 HTTP 8090 與相機均維持關閉。
- 未在 Intel／Qualcomm 硬體量測；KCF 的 CPU 模組隔離測試已驗證不匯入 NVIDIA runtime。

## 清除舊追蹤實作（2026-11-16）

已移除舊追蹤 Python 模組、獨立 YAML、C++ bridge、專用測試、安裝腳本與專用 CUDA RGB→RGBA 縮放函式，並清除其編譯產物及 native 建置快取。建置入口僅保留 CUDA geometry，操作文件統一使用 KCF。

本次 `./verify.sh`：SDK 56 passed、application 545 passed。CUDA geometry 從乾淨快取建置成功。既有 `validation/` 與移入驗證文件保留作為歷史證據。
