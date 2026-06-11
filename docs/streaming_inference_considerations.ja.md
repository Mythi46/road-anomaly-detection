# ストリーミング映像入力時の推論性能検討メモ / Streaming Inference Performance Notes

## 1. 目的 / Purpose

**日本語**  
本メモは、車載カメラまたはウェアラブルカメラから映像をサーバへアップロードし、サーバ側で路面異常検出を行う場合の性能確認項目を整理するものである。v0 段階では、端末側で AI 推論を行うのではなく、端末は映像送信を担当し、サーバ側でフレーム抽出および推論を行う構成を前提とする。

**English**  
This memo summarizes the performance points to check when video from an in-vehicle camera or wearable camera is uploaded to a server and road anomaly detection is performed on the server side. For v0, the device is assumed to only send video, while frame extraction and AI inference are handled on the server.

## 2. 想定構成 / Assumed Architecture

**日本語**  
想定構成は以下の通りである。

```text
Camera / wearable device
    -> video upload / streaming
    -> server-side video receiver
    -> frame extraction
    -> detection inference
    -> result output
```

この構成では、評価対象は端末側の AI 性能ではなく、映像アップロード帯域、サーバ側のフレーム抽出速度、単フレーム推論時間、エンドツーエンド遅延、同時処理可能なストリーム数である。

**English**  
The assumed architecture is as follows.

```text
Camera / wearable device
    -> video upload / streaming
    -> server-side video receiver
    -> frame extraction
    -> detection inference
    -> result output
```

In this architecture, the main evaluation target is not AI performance on the device. The key points are upload bandwidth, server-side frame extraction speed, inference time per frame, end-to-end latency, and the number of concurrent video streams the server can handle.

## 3. 確認すべき性能指標 / Key Performance Metrics

**日本語**

| 指標 | 内容 |
|---|---|
| 入力解像度 | 720p / 1080p など |
| 動画ビットレート | 上り通信帯域に影響 |
| フレーム抽出間隔 | v0 は 1 fps を基本想定 |
| 単フレーム推論時間 | 1枚あたりの推論処理時間 |
| エンドツーエンド遅延 | 撮影から検出結果出力までの時間 |
| 同時ストリーム数 | 同時に処理できる映像本数 |
| CPU / GPU 使用率 | Cloud Run CPU で十分か、GPU が必要か判断 |

**English**

| Metric | Meaning |
|---|---|
| Input resolution | For example, 720p or 1080p |
| Video bitrate | Affects required upload bandwidth |
| Frame sampling rate | v0 assumes 1 fps as the baseline |
| Inference time per frame | Processing time for one extracted image |
| End-to-end latency | Time from capture to detection output |
| Concurrent streams | Number of video streams processed at the same time |
| CPU / GPU utilization | Used to judge whether Cloud Run CPU is enough or GPU is required |

## 4. v0 での基本方針 / v0 Policy

**日本語**  
v0 では、動画の全フレームを推論対象としない。サーバ側で一定間隔にサンプリングしたフレームのみを推論する。基本案は、サーバ側で 1 fps 程度にフレームを抽出し、Cloud Run CPU 上で推論する構成である。これにより、端末側のハードウェア要件を抑えつつ、モデルやしきい値をサーバ側で管理できる。

**English**  
For v0, we should not run inference on every video frame. The server should sample frames at a fixed interval and only run inference on those frames. The baseline plan is to sample around 1 fps on the server side and run inference on Cloud Run CPU. This keeps the device requirements low and allows the model and thresholds to be managed on the server.

## 5. 推論速度の考え方 / How to Think About Inference Speed

**日本語**  
推論速度は、単に「モデルが何 fps 出るか」だけでは判断できない。実際には、動画デコード、フレーム抽出、前処理、モデル推論、後処理、結果出力を分けて確認する必要がある。

```text
total latency
  = upload latency
  + video decode / frame extraction
  + preprocessing
  + model inference
  + postprocessing
  + result output
```

**English**  
Inference speed should not be judged only by model FPS. In practice, we need to separate video decoding, frame extraction, preprocessing, model inference, postprocessing, and result output.

```text
total latency
  = upload latency
  + video decode / frame extraction
  + preprocessing
  + model inference
  + postprocessing
  + result output
```

## 6. 処理能力の簡易見積もり / Simple Capacity Estimate

**日本語**  
例えば、1フレームのサーバ側処理時間が 0.5 秒の場合、単純計算では 1インスタンスあたり約 2 fps を処理できる。v0 で 1ストリームあたり 1 fps を抽出する場合、理論上は 1インスタンスで約 2ストリームを処理できる。ただし、実際には通信、デコード、I/O、同時実行制御の余裕を考慮する必要がある。

```text
required_fps = number_of_streams × sampled_fps_per_stream

required_instances
  = required_fps / fps_per_instance
```

**English**  
For example, if server-side processing takes 0.5 seconds per frame, one instance can process about 2 fps in a simple estimate. If v0 samples 1 fps per stream, one instance could theoretically process about two streams. In practice, we still need margin for network, decoding, I/O, and concurrency control.

```text
required_fps = number_of_streams × sampled_fps_per_stream

required_instances
  = required_fps / fps_per_instance
```

## 7. 初回ストリーミングテストで記録する項目 / Items to Record in the First Streaming Test

**日本語**

| 項目 | 記録内容 |
|---|---|
| 入力動画 | 解像度、fps、codec、bitrate |
| 通信状態 | 上り帯域、切断有無、遅延 |
| 抽出設定 | 何 fps でフレーム抽出したか |
| 推論時間 | 平均、p95、最大 |
| エンドツーエンド遅延 | 平均、p95、最大 |
| サーバ負荷 | CPU、memory、同時リクエスト数 |
| 検出結果 | 検出数、誤検出傾向、未検出傾向 |

**English**

| Item | What to record |
|---|---|
| Input video | Resolution, fps, codec, bitrate |
| Network condition | Upload bandwidth, disconnections, delay |
| Sampling setting | Frame extraction rate |
| Inference time | Average, p95, maximum |
| End-to-end latency | Average, p95, maximum |
| Server load | CPU, memory, concurrent requests |
| Detection output | Number of detections, false-positive tendency, missed-detection tendency |

## 8. 技術上の注意点 / Technical Notes

**日本語**  
v0 では、30 fps の動画を全フレーム推論する必要はない。路面巡回では 1 fps 程度でも数メートルから十数メートル間隔で画像が得られるため、初期検証には十分である。また、検出精度が悪い場合でも、原因がモデル性能とは限らない。映像圧縮、ブレ、露出、カメラ角度、通信遅延、フレーム抽出品質を分けて確認する必要がある。

**English**  
For v0, it is not necessary to run inference on all frames of a 30 fps video. For road inspection, sampling around 1 fps can still provide images every several to around ten-plus meters, which is enough for initial validation. Also, if detection quality is poor, the cause may not be the model alone. We should separately check compression, motion blur, exposure, camera angle, network delay, and frame extraction quality.

## 9. 端末側推論について / On-device Inference

**日本語**  
車載端末やウェアラブル端末で推論を行う場合、端末スペック、発熱、電源、モデル更新、障害対応が追加課題となる。v0 ではサーバ側推論を優先し、端末側は映像送信に限定する方がリスクが低い。

**English**  
If inference is performed on an in-vehicle or wearable device, additional issues appear: device specification, heat, power supply, model updates, and failure handling. For v0, server-side inference is lower risk, while the device should only be responsible for sending video.

## 10. 現時点の結論 / Current Conclusion

**日本語**  
v0 では、車載またはウェアラブル端末が映像をサーバへアップロードし、サーバ側で 1 fps 程度に抽出したフレームを推論する構成が現実的である。初回テストでは、まず単一ストリームの処理時間を測定し、その後に同時ストリーム数を増やして処理能力を確認するのが望ましい。

**English**  
For v0, the realistic architecture is for the in-vehicle or wearable device to upload video to the server, and for the server to sample frames at around 1 fps and run inference. In the first test, we should measure the processing time for a single stream first, then increase the number of concurrent streams to evaluate capacity.
