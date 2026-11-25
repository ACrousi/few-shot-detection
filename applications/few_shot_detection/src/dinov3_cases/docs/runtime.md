# 執行流程、排程與影像輸出

[回程式導覽](../README.md) · [3D 深度框](depth-bbox.md)

## 模型耗時 topic 與網頁

網頁 Live view 顯示各階段最近一次成功完成的毫秒數。
`/api/status` 的 `model_timings` 與 ROS topic `/medicine_recognition/model_timings`
共用量測資料。topic 使用 `std_msgs/msg/String` JSON，5 Hz、Reliable、KeepLast(1)、Volatile。

```bash
ros2 topic echo /medicine_recognition/model_timings std_msgs/msg/String
```

topic 的 JSON 只有三個數值欄位，單位固定為 ms：

```json
{"kcf_ms": 23.3, "dinov3_ms": 205.7, "sam3_ms": 219.8}
```

`/api/status` 的 `model_timings` 使用相同格式。尚無成功樣本時數值為 `null`，
網頁顯示 `—`；快取結果與執行中保留最近成功值。
來源或設定改變時清空，忽略舊工作的遲到結果。

計時均為 wall time：KCF 是目前 live contexts 整體追蹤更新（含 CPU 前處理，
不含背景 candidate 初始化／補追）；DINOv3 是整批圖庫刷新、辨識及診斷資料建立；
SAM3 是 `worker_detect_ms`（不含主程序 IPC）。這些不是純 GPU kernel 時間，
非同步階段不可直接相加作為總延遲。KCF 在所有偵測間隔下都執行。

## 必須維持的生命週期

- SAM3 框屬於較舊的影格；須在偵測原圖初始化，再補追至當前影格，才可交接。大幅跳幀仍可能追錯。
- 相同影格重算不能增加追蹤命中次數或重複累加平滑樣本。
- 相機持續送影格時，SAM3／DINO／3D 的背景完成通知合併到下一張 RGB，避免同影格再複製、追蹤、發布及畫圖編碼。若超過 200 ms 沒有新輸入，仍可刷新該影格，讓首次 3D 與延後類別完成；靜態上傳可立即套用背景結果。這類刷新不增加 processed RGB FPS。
- KCF 沿用分類，新的辨識結果只依 seed ownership 更新仍存活物件。
- 診斷裁切屬於辨識當時的影格，不可套用後續 KCF 框冒充目前影像。
- 共享影格在所有租約釋放前不可覆寫；舊 generation 的背景結果不可發布。
- 診斷快照保留完整舊圖庫版本，避免更新圖庫時混用特徵與標籤。


網頁 `processed RGB FPS`（API `result_fps`）只計不同 `(generation, frame_id)` 的已完成 RGB，不把同影格的背景刷新算成新影格；相機的 `input FPS` 是輸入回呼頻率。主迴圈只保留可配對的深度快照並提交背景 3D 工作，不等待深度或量測完成；非同步模式的 DINOv3 在背景執行，`recognize_ms` 記錄交接結果的背景辨識耗時；細分計時和影像輸出方式見 [建置與執行說明](../../../BUILD_GUIDE.md)。

啟動或套用設定後，第一份框等待 SAM3 與 KCF 初始化完成便開始持續追蹤，DINO 類別稍後更新；相機擷取與網頁預覽仍獨立運作。可配對深度會在等待推論前保留。這可避免首次初始化等待期間持續補追而反覆超限；模型載入完成後，網頁會顯示準備首次偵測的狀態。`/api/status` 的 `grounding_schedule` 可查看目前初始化、背景工作及最近一次補追丟棄原因。

### 共用 ROS Image 即時影像

單一疊圖輸出為 `/medicine_recognition/image_annotated`（`sensor_msgs/msg/Image`，
`bgr8`、960×540、Reliable、KeepLast(1)、Volatile），保留來源 RGB 的 header。
`annotated_image_topic` 可覆寫 topic；`video_fps` 控制最大輸出頻率（1–30）。
疊圖與 ROS 發布各用獨立 worker，待處理工作只保留最新一份，不讓慢速 DDS 阻塞推論。
無網頁觀看者且無外部 Image 訂閱者時不繪圖。

首頁與圖庫頁都使用 `<img src="/image_annotated.mjpg">`。HTTP bridge 實際訂閱同一個
annotated Image topic，背景 worker 將最新影格轉為 JPEG（quality=90），以 MJPEG 提供給網頁。
所有觀看者共用同一份 JPEG；每個接收端只取最新完成影格，不建立逐幀佇列。
無網頁觀看者時不做 JPEG 轉換；靜態來源可重送快取 JPEG，無需重新編碼。
連線關閉或輸出錯誤會回收觀看者；圖片播放器會在網路錯誤後重連。
HTTP 經反向代理時應停用此路徑的 response buffering。

3D 模式等待有效深度量測及相機參數，不暫時畫 2D 框；DINO 尚未完成時可先顯示
已就緒 3D 框和 SAM3 名稱。相機辨識結果超過 3 秒或無可見框時顯示原始相機預覽；
上傳圖片不過期，沒有 ROS header 時使用繪圖當下節點時間。
圖庫手動擷取仍使用原始 RGB，ROI 預覽依 `<img>` 的 natural size 及顯示留白換算。

RViz2 使用 **Image** display，選擇上述 topic，Transport Hint=`raw`、Reliability=`Reliable`，
Queue Size 建議為 1。結構化 Detection2D／Detection3D、AprilTag topic、物件診斷及圖庫圖片保持原有用途。

網頁圖片串流已取代先前的影片傳輸；舊影片封包 topic、協商／peer API 及編碼設定已移除。
MJPEG 與 raw Image 的頻寬可能較高；實際 CPU、頻寬及端到端延遲需在目標場景量測。

## KCF 追蹤與交接

每個物件使用獨立的 CPU OpenCV KCF。追蹤失敗立即隱藏框，`tracker_max_missed_seconds` 保留 metadata 與公開 ID 至寬限結束。失追會提前要求 SAM3；仍有隱藏物件時，重試間隔為 `min(grounding_detection_interval, tracker_max_missed_seconds / 3)`，且只在新影格及背景空閒時提交。

候選 tracker 在 SAM3 原圖初始化，再補追最新影格，與現有軌跡在共同影格幾何配對。配對者交由新 context 接手；SAM3 漏偵測但 KCF 仍有效者保留。分類依 seed ownership 延後回填，不回退目前框。沒有長期 ReID 保證。

`grounding_schedule` 提供 `candidate_create_ms`、`candidate_seed_ms`、`candidate_catchup_ms`、`geometry_ready_ms` 及 `job_ms`。DINO 失敗保留框，錯誤見 `recognition_error`。完整參數見 [KCF 說明](../../../../../docs/kcf-tracking.md)。
