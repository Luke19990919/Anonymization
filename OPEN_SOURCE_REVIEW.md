# 開源技術評估（2026-11-26）

本次已修復既有程式中的欄位標籤與幾何定位缺口；未以更換模型取代修復。以下是已讀取來源與授權的候選，不能視為已完成整套安裝或私人影片驗收。

| 候選 | 本次查得的能力／授權 | 可借用部分與邊界 |
| --- | --- | --- |
| [OpenScrub](https://github.com/austinmabry/OpenScrub) | 本機影片文字去識別、追蹤、出現邊界回溯；[Apache-2.0](https://github.com/austinmabry/OpenScrub/blob/main/LICENSE) | 優先評估它的分析／追蹤／輸出分層及 onset 回溯；README 自述 best-effort，不能照搬其抽樣設定或宣稱零漏遮。 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 中文與多語文字定位／辨識、結構化輸出；[Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE) | 適合作為替代或獨立交叉驗證 OCR；未在本機完成該模型基準，不宣稱比現用後端快或更準。它不是隱私判定或通知追蹤的完整解。 |
| [Presidio](https://github.com/data-privacy-stack/presidio) | 可擴充的文字／圖片個資分析與遮蔽；[MIT](https://github.com/data-privacy-stack/presidio/blob/main/LICENSE) | 適合作為規則／辨識器介面參考；不是現成的逐幀影片與繁中姓名保證。原 microsoft/presidio 網址目前轉向 data-privacy-stack。 |

已讀取 OpenScrub 的 openscrub.py 中追蹤與 smooth_dense_tracks／onset 回溯的程式；本版沒有直接複製其程式碼，也沒有新增 PaddleOCR 或 Presidio 相依性。未經驗證的套件不會直接取代現有後端。若後續複製程式或散布模型，必須保留適用授權、著作權及模型／相依項目的個別聲明。

目前現用的 EasyOCR 與 Privacy Filter 本來就是可檢查原始碼的元件。第二支失敗影片的首字問題在本專案的欄位政策與定位，不是缺少一個大型模型。

跨模型維護模板在 templates/maintenance-agent.md；結構化判定模板與 schema 在 templates/privacy-classifier.md 及 schemas/privacy-findings.schema.json。它們不會自行傳送私人畫面到外部 LLM，也不會啟用常駐修復代理。
