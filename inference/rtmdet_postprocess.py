import numpy as np

class RTMDetInsPostProcessor:
    def __init__(self, num_classes=14, score_threshold=0.3, iou_threshold=0.45):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.strides = {100: 8, 50: 16, 25: 32}

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _get_layer_by_shape(self, data_dict, expected_size, expected_channels):
        for name, data in data_dict.items():
            if data.shape[1] == expected_size and data.shape[3] == expected_channels:
                return data
        return None

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _apply_nms(self, candidates):
        if not candidates: return []
        # スコア順にソート
        candidates.sort(key=lambda x: x['score'], reverse=True)
        keep = []
        while candidates:
            best = candidates.pop(0)
            keep.append(best)
            candidates = [c for c in candidates if self._iou(best['box'], c['box']) < self.iou_threshold]
        return keep

    def process(self, raw_data, meta):
        '''
        print("--- デバッグ: 出力レイヤー一覧 ---")
        for name in raw_data.keys():
            print(f"Layer: {name} | Shape: {raw_data[name].shape}")
        print("-------------------------------")
        '''
        
        # 1. デ量子化
        dequantized_data = {}
        for name, data in raw_data.items():
            zp = meta[name]['zp']
            scale = meta[name]['scale']
            dequantized_data[name] = (data.astype(np.float32) - zp) * scale

        all_candidates = []

        # 2. 各解像度を走査
        for size, stride in self.strides.items():
            scores_layer = self._get_layer_by_shape(dequantized_data, size, self.num_classes)
            boxes_layer = self._get_layer_by_shape(dequantized_data, size, 4)
            # ★マスク係数レイヤーを取得（チャンネル数はモデルに合わせて32など調整）
            masks_layer = self._get_layer_by_shape(dequantized_data, size, 169) 

            if scores_layer is None or boxes_layer is None:
                continue

            for y in range(size):
                for x in range(size):
                    logits = scores_layer[0, y, x, :]
                    probs = self._sigmoid(logits)
                    max_score = np.max(probs)

                    if max_score > self.score_threshold:
                        class_id = np.argmax(probs)
                        dist = boxes_layer[0, y, x, :] * stride 
                        
                        center_x = x * stride
                        center_y = y * stride
                        
                        box = [center_x - dist[0], center_y - dist[1],
                               center_x + dist[2], center_y + dist[3]]
                        
                        # ★ここで各候補に mask_coeff を持たせる
                        # masks_layer が存在する場合のみ取得
                        coeff = masks_layer[0, y, x, :] if masks_layer is not None else None
                        
                        all_candidates.append({
                            "box": box,
                            "score": float(max_score),
                            "class_id": int(class_id),
                            "mask_coeff": coeff # ←ここに保存！
                        })

        # 3. NMS実行
        final_results = self._apply_nms(all_candidates)
        
        # ★Protoレイヤーを別途取り出しておく（あとで計算に使うため）
        # RTMDetのProtoは通常 160x160 などのサイズで 8チャンネル
        proto = self._get_layer_by_shape(dequantized_data, 100, 8) # サイズはモデルによる
        
        return final_results, proto # protoも一緒に返すと後で楽です
    

class RTMDetPostProcessor:
    def __init__(self, num_classes=14, score_threshold=0.3, iou_threshold=0.45):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.strides = {100: 8, 50: 16, 25: 32}

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _get_layer_by_shape(self, data_dict, expected_size, expected_channels):
        for name, data in data_dict.items():
            if data.shape[1] == expected_size and data.shape[3] == expected_channels:
                return data
        return None

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _apply_nms(self, candidates):
        if not candidates:
            return []
        candidates.sort(key=lambda x: x['score'], reverse=True)
        keep = []
        while candidates:
            best = candidates.pop(0)
            keep.append(best)
            candidates = [c for c in candidates if self._iou(best['box'], c['box']) < self.iou_threshold]
        return keep

    def process(self, raw_data, meta):
        '''
        print("--- デバッグ: 出力レイヤー一覧 ---")
        for name in raw_data.keys():
            print(f"Layer: {name} | Shape: {raw_data[name].shape}")
        print("-------------------------------")
        '''
        
        # 1. デ量子化
        dequantized_data = {}
        for name, data in raw_data.items():
            zp = meta[name]['zp']
            scale = meta[name]['scale']
            dequantized_data[name] = (data.astype(np.float32) - zp) * scale

        all_candidates = []

        # 2. 各解像度を走査（スコアレイヤーとボックスレイヤーのみ使用）
        for size, stride in self.strides.items():
            scores_layer = self._get_layer_by_shape(dequantized_data, size, self.num_classes)
            boxes_layer  = self._get_layer_by_shape(dequantized_data, size, 4)

            if scores_layer is None or boxes_layer is None:
                continue

            for y in range(size):
                for x in range(size):
                    logits = scores_layer[0, y, x, :]
                    probs = self._sigmoid(logits)
                    max_score = np.max(probs)

                    if max_score > self.score_threshold:
                        class_id = np.argmax(probs)
                        dist = boxes_layer[0, y, x, :] * stride

                        center_x = x * stride
                        center_y = y * stride

                        box = [
                            center_x - dist[0],
                            center_y - dist[1],
                            center_x + dist[2],
                            center_y + dist[3]
                        ]

                        all_candidates.append({
                            "box": box,
                            "score": float(max_score),
                            "class_id": int(class_id),
                        })

        # 3. NMS実行
        final_results = self._apply_nms(all_candidates)

        return final_results