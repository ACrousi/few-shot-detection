# 3D 深度框與背景量測

[回程式導覽](../README.md) · [執行流程](runtime.md)

### CPU RANSAC 平面旋轉框

`geometry/depth_bbox.py` 沿用 isaac_ros-dev 的主平面流程。先依中央深度選取前景連通區，
用相機內參反投影，套用深度分位數與 CloudFilter，再以 Open3D CPU
`PointCloud.segment_plane` 執行 RANSAC。距離門檻預設 0.005 m，
每次取 3 點、128 次迭代、成功機率 0.999。主平面需至少涵蓋一半輸入點，
且內點數達到 `bbox3d_min_points`。

主平面內點的最小面積矩形長邊決定局部 X，平面法向量決定局部 Z。
原連通點雲在此座標系的分位數上下界決定中心與尺寸。
平面估計失敗時使用相機座標軸，並回報 `camera_aligned_fallback:<原因>`。

成功時 `method` 為 `central_depth_component_plane_rectangle`，
回退時為 `central_depth_component_mean_z`。診斷包含
`plane_backend: open3d_cpu`、`plane_inlier_count`、`plane_threshold`。
反投影、RANSAC、旋轉框與投影校正使用 NumPy、OpenCV、Open3D 與 SciPy 的 CPU 運算。
單視角點雲的深度尺寸描述可見表面變化。

### 平面偵測前的點雲篩選

節點使用深度分位數篩選後的樣本計算覆蓋率及深度中位數。

- `bbox3d_min_cloud_coverage: 0.15`：quantile 後點數 / 裁切後 bbox 像素數，低於 15% 的碎片直接拒絕，重複出現也不放行。
- `bbox3d_min_coverage_ratio: 0.35`：覆蓋率低於同物件上一筆接受值的 35% 時視為可疑。使用比例避免 2D bbox 大小變更直接影響點數比較。
- `bbox3d_max_depth_jump: 0.10`：quantile 後中位深度與上一筆接受值相差超過 10 公分時視為可疑。
- `bbox3d_depth_confirmations: 3`：可疑新區塊需連續 3 個不同深度時間戳支持；新候選間深度差需在跳動門檻的一半以內，覆蓋率彼此至少一半。回復原區塊、低覆蓋或量測失敗會中斷確認。

拒絕狀態為 `insufficient_cloud_coverage` 或 `cloud_change_pending:<次數>/<需求>`。
通過檢查的量測提供 `plane_input_count`、`plane_inlier_count`、`plane_backend`、`cloud_coverage` 與 `cloud_depth` 作為 HTTP 診斷。
背景 worker 依物件追蹤身分保存量測歷史。物件消失或來源變更時，對應歷史隨之清除。
獨立呼叫 `estimate_bbox3d` 時，傳入 `CloudFilter` 和每個物件獨立的 `cloud_state` 即可套用覆蓋與連續性檢查。

### 使用 2D 偵測框補足可見範圍

深度缺測、前景連通區只包含部分物體，以及分位數裁邊，都可能讓點雲框的投影偏離 2D 偵測框。
即時流程保留 RANSAC 旋轉框的旋轉、中心相機 Z 與局部 Z 尺寸，調整相機 X/Y 中心和局部 X/Y 尺寸，使八個頂點的投影邊界貼近同一量測影格的偵測框。
ROS Detection3D 與 3D 疊圖使用校正後的框。校正後的橫向尺寸同時取決於點雲與 2D 偵測框。獨立呼叫 `estimate_bbox3d` 可取得純點雲量測。

HTTP 的 `measured_position` 與 `measured_size` 保留點雲框。像素診斷包含 `fit_initial_error_px` 及接受校正時的 `fit_error_px`。
最大邊界誤差在原圖 0.25 px 內時標記 `matched`。有改善但仍超過門檻時標記 `approximate`。求解失敗或誤差沒有改善時保留點雲量測。

### 背景 3D 框排程

每張新處理的 RGB 影格都提交背景 3D 工作，與 `grounding_detection_interval` 和 `video_fps` 無關。
同一影格因 SAM3 交接或背景完成而重跑時不重複提交。
背景涵蓋深度轉換、點雲分割、CPU RANSAC，以及旋轉框 中心和尺寸計算；只有一個執行中的工作與一份最新待算工作，忙碌時覆蓋舊的待算工作。
RGB 主迴圈不等待深度到達；它保留當下可配對的深度，若推論後才有對應深度則再取一次快照。

繪圖與 ROS 發布前會讀取最新完成的一整份量測，不混搭不同次的中心、尺寸及旋轉。
`bbox3d_max_age` 直接使用正秒數，從來源影格接收時間起算。目前部署為 1.0 秒，程式預設為 1.5 秒；不隨 `video_fps` 變動，因此影像停止到達時舊結果仍會過期。首次尚未完成或結果過期時不畫舊框；量測失敗可暫用尚未過期的有效快取，不延長原有效期限。
背景結果透過相鄰 2D 框的幾何配對維持內部物件身分；順序更換與部分遺失不會直接以陣列索引套用舊結果，
消失的身分不再使用，來源 generation、影像大小或座標框架變更時清空快取。
這仍有相近物件交叉或遮擋的配對歧義，尚未加入外觀式的持久物件識別。

`bbox3d_source` 記錄來源 RGB 影格、座標框架和時間戳，`bbox3d_age_seconds` 記錄量測年齡。
ROS 每筆 `Detection3D.header` 保留該量測的來源時間，陣列 header 使用其中最新的量測時間；
2D 辨識與疊圖影像仍使用各自目前影格的時間。
`/tag_detections` 借用 AprilTag 訊息格式發布物件資料；每筆 `pose` 的 XYZ 與 quaternion
沿用 `Detection3D` 的框中心及旋轉，`pose.header` 保留來源量測時間與座標系。
陣列 header 與水平／垂直的 2D `corners` 仍對應目前影格。
朝向由可見點雲估算，遇到近似對稱的點雲時可能有方向歧義。
缺少旋轉的舊資料也使用單位 quaternion。Covariance 未估算，保留訊息預設值。
`/api/status` 的 `performance.bbox3d_schedule` 顯示獨立間隔與背景量測耗時；`bbox3d_ms` 則是前景提交／讀取快取的耗時。
參數在啟動時讀入，修改 YAML 後需重啟節點。

