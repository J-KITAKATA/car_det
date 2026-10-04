import cv2
import numpy as np
from hailo_runner import HailoInferenceRunner
from rtmdet_postprocess import RTMDetInsPostProcessor
import sys
import time

class VehicleApp:
    def __init__(self, model_path, score_thr=0.45):
        self.class_names = [
            "Kei_car", "Sedan", "Coupe", "SUV", "Minivan", "Small_Truck",
            "Large_Truck", "Bus", "Scooter", "Compact_car", "Station_Wagon",
            "Commercial_Van", "Hatchback", "Motorcycle"
        ]
        self.class_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
            (64, 64, 64), (192, 192, 192)
        ]
        
        self.runner = HailoInferenceRunner(model_path)
        self.post_processor = RTMDetInsPostProcessor(num_classes=14, score_threshold=score_thr)
        self.target_h, self.target_w = self.runner.input_shape[:2]

    def process_frame(self, frame):
        """1枚のフレームに対する共通の推論・マスク処理"""
        if frame is None: return None
        
        orig_h, orig_w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h))

        # 推論
        raw_data, meta = self.runner.run(img_resized)
        results, proto = self.post_processor.process(raw_data, meta)

        scale_x, scale_y = orig_w / self.target_w, orig_h / self.target_h
        mask_img = frame.copy()
        draw_img = frame.copy()

        for res in results:
            box = res['box']
            score = res['score']
            cls_id = res['class_id']
            mask_coeff = res.get('mask_coeff')

            x1 = int(max(0, box[0] * scale_x))
            y1 = int(max(0, box[1] * scale_y))
            x2 = int(min(orig_w, box[2] * scale_x))
            y2 = int(min(orig_h, box[3] * scale_y))
            color = self.class_colors[cls_id % len(self.class_colors)]

            if mask_coeff is not None and proto is not None:
                weights = mask_coeff[:16]
                p_layer = proto[0]
                combined = np.zeros((100, 100), dtype=np.float32)
                for i in range(8):
                    w = weights[i] + weights[i+8] if len(weights) > i+8 else weights[i]
                    combined += p_layer[:, :, i] * w

                mask = 1 / (1 + np.exp(-combined * 2.0))
                px1, py1 = max(0, int(box[0] / 8)), max(0, int(box[1] / 8))
                px2, py2 = min(100, int(box[2] / 8)), min(100, int(box[3] / 8))

                if (px2 - px1) > 0 and (py2 - py1) > 0:
                    mask_crop = mask[py1:py2, px1:px2]
                    bw, bh = x2 - x1, y2 - y1
                    mask_resized = cv2.resize(mask_crop, (bw, bh))
                    
                    # --- 安定化ロジック ---
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
                    mask_final_raw = cv2.dilate(mask_cleaned, kernel, iterations=1)

                    contours, _ = cv2.findContours(mask_final_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    final_mask_canvas = np.zeros_like(mask_final_raw)
                    if contours:
                        for cnt in contours:
                            if cv2.contourArea(cnt) > (bw * bh * 0.008):
                                cv2.drawContours(final_mask_canvas, [cnt], -1, 1, thickness=cv2.FILLED)
                        
                        mask_final = final_mask_canvas > 0
                        roi = mask_img[y1:y2, x1:x2]
                        if roi.shape[0] == mask_final.shape[0] and roi.shape[1] == mask_final.shape[1]:
                            roi[mask_final] = color
                            mask_img[y1:y2, x1:x2] = roi

            label_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"ID:{cls_id}"
            full_label = f"{label_name} {score:.2f}"
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(draw_img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(draw_img, full_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, 0.4, draw_img, 0.6, 0)

    # FPS描画
    def _draw_fps(self, img, fps):
        """FPS値を画像の右上に描画する"""
        label = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        img_h, img_w = img.shape[:2]
        margin = 10

        # 右上の座標を計算
        x = img_w - text_w - margin
        y = text_h + margin

        # 背景ボックス（視認性のため）
        cv2.rectangle(img,
                    (x - 4, y - text_h - 4),
                    (x + text_w + 4, y + baseline + 2),
                    (0, 0, 0), cv2.FILLED)

        # テキスト描画
        cv2.putText(img, label, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
        return img

    # --- モード別実行メソッド ---
    def run_image(self, src, output):
        print(f"--- 実行モード: Image ---")
        img = cv2.imread(src)
        result = self.process_frame(img)
        cv2.imwrite(output, result)
        print(f"結果を保存しました: {output}")

    def run_video(self, src, output):
        print(f"--- 実行モード: Video ---")
        cap = cv2.VideoCapture(src)
        
        # 保存設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            result = self.process_frame(frame)
            out.write(result)
            cv2.imshow('Hailo Video Processing', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"動画を保存しました: {output}")

    def run_live(self):
        print(f"--- 実行モード: Live (Webカメラ) ---")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # V4L2バックエンドを明示的に指定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 810)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # ウォームアップ
        for _ in range(10):
            cap.read()

        fps = 0.0
        prev_time = time.perf_counter()  # ① 高精度タイマーで初期化

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            # ③ FPS計測（1フレームごとの経過時間から算出）
            curr_time = time.perf_counter()
            elapsed = curr_time - prev_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            prev_time = curr_time

            # ④ 右上にFPS描画
            result = self._draw_fps(result, fps)

            cv2.imshow('Hailo Live Camera', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 【実行設定】
    # ---------------------------------------------------------
    model_path = r'../models/rtm_s_car800.hef'
    
    mode = "live"  # "image", "video", "live"
    src = r'../test_images/IMG_0096_task3.JPG' # Liveモードの場合は無視されます
    output = r'./result.jpg' # srcが動画の場合は ./result.mp4 などに変更してください
    # ---------------------------------------------------------

    # モードと許可される拡張子の対応表
    EXTENSIONS = {
        "image": ('.jpg', '.jpeg', '.png', '.bmp'),
        "video": ('.mp4', '.avi', '.mov', '.mkv')
    }

    app = VehicleApp(model_path)

    # 1. 画像モードのバリデーション
    if mode == "image":
        if not src.lower().endswith(EXTENSIONS["image"]):
            print(f"エラー: 画像モードですが、不適切な拡張子です。許可: {EXTENSIONS['image']}")
            sys.exit()
        app.run_image(src, output)

    # 2. 動画モードのバリデーション
    elif mode == "video":
        if not src.lower().endswith(EXTENSIONS["video"]):
            print(f"エラー: 動画モードですが、不適切な拡張子です。許可: {EXTENSIONS['video']}")
            sys.exit()
        app.run_video(src, output)

    # 3. ライブモード（バリデーション不要）
    elif mode == "live":
        check_cap = cv2.VideoCapture(0)
        is_opened = check_cap.isOpened()
        check_cap.release()

        if not is_opened:
            # raise ConnectionError の代わりに print + sys.exit
            print("\n" + "="*50)
            print("【お知らせ】有効なカメラデバイスが見つかりませんでした。")
            print("・カメラが物理的に接続されているか確認してください。")
            print("・他のアプリ（Zoomやブラウザ等）がカメラを使っていないか確認してください。")
            print("="*50 + "\n")
            sys.exit() # ここでプログラムを綺麗に終了させる
        
        print("カメラの接続を確認しました。ライブ配信を開始します...")
        app.run_live()

    else:
        print("エラー: 未定義のモードです。'image', 'video', 'live' のいずれかを指定してください。")
        sys.exit()