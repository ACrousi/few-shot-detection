# Robotic Suite

以平台為單位執行完整 application pipeline。平台自行管理前處理、模型、後處理、
記憶體、資料傳輸與資源釋放；suite 提供設定讀取、啟動及環境管理。

```text
rsctl / application → platforms/<platform>/pipeline.py → 完整計算流程
```

沒有共用 backend、tensor schema、stage provider 或抽象基底類別。
共用 SDK 現在只保留 YAML／程序啟動工具，不傳遞模型中間張量。

```bash
python -m pip install -e 'packages/robotic_inference_sdk[test]' -e tools/rsctl
python -m pytest -q
rsctl init --name my_demo --destination generated
python -m pip install -e generated/my_demo
rsctl validate --config generated/my_demo/config/default.yaml
rsctl pipeline-run --config generated/my_demo/config/default.yaml
```

產生的 CPU 範例只計算 mean RGB，明確標示 MOCK。NVIDIA／Intel／Qualcomm 的範本入口
尚待應用實作，直接執行會報錯。few-shot-detection 有自己的真實 NVIDIA pipeline。

- [架構與移轉說明](docs/architecture.md)
- [應用開發](docs/application_development.md)
- [環境管理](docs/environments.md)

舊 InferenceBackend、ModelConfig、registry 與 PlatformProfile API 已移除；使用者需改用
平台自有程式。Runtime 版本與硬體相容性由各平台 pipeline 驗證。
