import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, 
                            FormatType)

class HailoInferenceRunner:
    def __init__(self, hef_path):
        # 1. HEFファイルのロード
        self.hef = HEF(hef_path)
        self.hef_path = hef_path
        
        # 2. 出力レイヤーのメタデータ（ZP, Scale, Shape）を自動取得して保持
        self.output_meta = {}
        for info in self.hef.get_output_vstream_infos():
            self.output_meta[info.name] = {
                'shape': info.shape,
                'zp': info.quant_info.qp_zp,
                'scale': info.quant_info.qp_scale
            }
        
        # 入力情報の取得（リサイズ確認用）
        input_vstream_infos = self.hef.get_input_vstream_infos()
        self.input_info = input_vstream_infos[0]

        self.input_name = self.input_info.name
        self.input_shape = self.input_info.shape

    def run(self, input_image):
        """
        input_image: NumPy配列 (H, W, 3) - UINT8形式
        returns: (raw_data_dict, meta_dict)
        """
        # 3. VDevice (チップ) の作成と構成
        target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe)
        
        # ネットワークグループの構成
        network_groups = target.configure(self.hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        # 入出力ストリームの設定 (UINT8のままやり取りする設定)
        input_vstreams_params = InputVStreamParams.make(
            network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.UINT8)

        # 推論の実行
        input_data = {self.input_info.name: np.expand_dims(input_image, axis=0)}
        
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                # チップからのUINT8生データを取得
                raw_results = infer_pipeline.infer(input_data)
        
        target.release() # デバイスの解放
        
        # 生の推論結果辞書と、それを変換するためのメタデータを返す
        return raw_results, self.output_meta