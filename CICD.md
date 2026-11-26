# 持續整合與發布

此資料夾須作為 GitHub 儲存庫根目錄；目前只交付流程檔，未連接或推送任何遠端儲存庫，GitHub 執行結果未經證實。

`.github/workflows/ci.yml`：push／PR 執行 Python 3.11 測試、Node 22 時間估算檢查及 wheel／sdist 打包，保存建置產物。使用 Linux CPU runner，不下載模型權重，不跑真實模型影片測試。相依安裝需要網路，初次安裝較慢。

`.github/workflows/release.yml`：維護者推送 v 開頭的版本標籤後，重新測試、建置並建立 GitHub Release。先讓標籤與 pyproject.toml 版本一致。GitHub Release 包含 wheel／sdist，並非免安裝應用程式或網站部署；目前 ZIP 分享包另行製作。沒有自動上傳私人影片、模型權重或本機暫存資料。

正式採用前應設定分支／標籤保護、鎖定完整相依版本與工作流程 action 的 commit SHA，並在自有 macOS GPU runner 加入虛構影片覆蓋率及效能驗證。Windows 啟動器仍待實機驗證。測試通過不能代表未知錄影沒有漏遮。

正式 Release 現在必須通過 scripts/check_release.py；有已知漏遮／誤遮、缺少目前程式的保留集證據時拒絕發布。修復代理與獨立驗收的權限隔離尚未接通，不能以本機檔案存在宣稱自主維護已啟用。詳見 MAINTENANCE.md。
