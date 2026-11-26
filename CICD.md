# CI/CD 執行與界線

來源儲存庫：https://github.com/Luke19990919/Anonymization 。修復透過 codex/fix-first-character-redaction 與草稿 PR 交付，不自動合併或發布已知失敗版本。

- ci.yml 在 push／PR／手動執行測試、Node ETA、wheel／sdist 建置，另設 privacy_acceptance job 阻擋尚未通過獨立隱私驗收的版本。
- model-regression.yml 在相關程式變更或手動啟動時，用公開權重與虛構產生器執行實際 OCR／NLP 回歸。只上傳允許的數值 JSON 與 CI 自行產生的虛構通知影片，不上傳私人影片。公共模型以版本化鍵快取。
- release.yml 在 v 標籤執行測試與正式驗收閘門；閘門不通過，不建立正式 Release。

初次實際執行：提交 783444f 的 Linux 測試、ETA、建置及 artifact 通過；隱私閘門按預期失敗。macOS 首字表單實際模型回歸通過，通知回歸出現本機未重現的失敗；後續增加數值診斷及證據 artifact 追查，未放寬驗收。最新狀態以 Actions 與 PR 檢查為準。

verify_cycle.py 是可重跑的驗證器，不是常駐 AI 修復服務；repair_executor_connected 仍為 false。跨模型代理模板在 templates/maintenance-agent.md，權限隔離的修復執行器仍需另外接通。未宣稱已實現 RSI。私人媒體不進入公開 CI，也沒有用戶端自動更新。
