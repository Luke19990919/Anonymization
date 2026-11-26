# 第三方元件

本專案自行提供影片串接、設定、遮罩與示範程式。安裝器會從各元件來源安裝相依套件；原始模型權重不包含於分享壓縮檔。

| 元件 | 用途 | 上游授權與來源 |
|---|---|---|
| OpenAI Privacy Filter | 文字個資分類 | Apache-2.0；[程式](https://github.com/openai/privacy-filter)、[模型](https://huggingface.co/openai/privacy-filter) |
| EasyOCR | 繁中與英文 OCR | Apache-2.0；[官方專案](https://github.com/JaidedAI/EasyOCR) |
| PyAV | 影片解碼與編碼介面 | BSD-3-Clause；[官方專案](https://github.com/PyAV-Org/PyAV) |
| imageio-ffmpeg | 提供 FFmpeg 執行入口 | BSD-2-Clause 包裝器；[官方專案](https://github.com/imageio/imageio-ffmpeg)。FFmpeg 二進位及其編譯選項依套件附帶授權，可能涉及 LGPL／GPL。 |
| Noto Sans TC | 繁中文字型 | SIL Open Font License 1.1；[官方字型來源](https://github.com/google/fonts/tree/main/ofl/notosanstc)，完整授權隨字型置於套件 assets/OFL.txt。 |

PyTorch、TorchVision、Pillow、NumPy、OpenCV 等相依套件保留各自的授權。分享或再發佈包含第三方二進位／權重的成品時，應一併保留其授權文件；本分享包不包含模型或執行環境。
