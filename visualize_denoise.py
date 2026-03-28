import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ==================== 绘图函数 ====================
def plot_audio_comparison(clean_path, noisy_path, denoised_path, streaming_path=None, save_path='assert/comparison.png'):
    """
    绘制干净语音、含噪语音、降噪后语音（整段推理）、降噪后语音（流式推理）的波形图和频谱图对比（2行4列）

    Parameters:
    - clean_path: 干净语音路径
    - noisy_path: 含噪语音路径
    - denoised_path: 整段推理降噪语音路径
    - streaming_path: 流式推理降噪语音路径（可选，若为None则只显示前三项）
    - save_path: 保存图片路径
    """
    # 加载音频（统一采样率 16kHz）
    clean, sr = librosa.load(clean_path, sr=16000)
    noisy, _ = librosa.load(noisy_path, sr=16000)
    denoised, _ = librosa.load(denoised_path, sr=16000)

    # 加载流式音频（如果提供）
    if streaming_path and os.path.exists(streaming_path):
        streaming, _ = librosa.load(streaming_path, sr=16000)
        audio_list = [clean, noisy, denoised, streaming]
        titles = ['干净语音', '含噪语音', '降噪后（整段推理）', '降噪后（流式推理）']
        n_cols = 4
    else:
        audio_list = [clean, noisy, denoised]
        titles = ['干净语音', '含噪语音', '降噪后（整段推理）']
        n_cols = 3
        if streaming_path:
            print(f"⚠️ 流式音频文件不存在：{streaming_path}，将忽略该项")

    # 确保长度一致（取最短长度）
    min_len = min(len(audio) for audio in audio_list)
    audio_list = [audio[:min_len] for audio in audio_list]
    time = np.linspace(0, len(audio_list[0]) / sr, num=len(audio_list[0]))

    # 创建图形（2行 n_cols 列）
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    fig.suptitle('语音降噪效果对比', fontsize=20, fontweight='bold', y=0.98)

    # ================== 第一行：波形图 ==================
    for i, (audio, title) in enumerate(zip(audio_list, titles)):
        # 波形图
        ax_wave = axes[0, i]
        ax_wave.plot(time, audio, linewidth=0.8, alpha=0.9)
        ax_wave.set_title(title + ' - 波形', fontsize=14)
        ax_wave.set_xlabel('时间 (秒)', fontsize=11)
        ax_wave.set_ylabel('幅值', fontsize=11)
        ax_wave.grid(alpha=0.3)
        # 不同音频使用不同颜色
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
        ax_wave.lines[0].set_color(colors[i % len(colors)])

        # ================== 第二行：频谱图 ==================
        ax_spec = axes[1, i]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                       ax=ax_spec, cmap='magma')
        ax_spec.set_title(title + ' - 频谱', fontsize=14)
        fig.colorbar(img, ax=ax_spec, format='%+2.0f dB')

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'✅ 对比图已保存至 {save_path}')


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 修改为实际的文件路径
    clean_wav = 'assert/D4_750.wav'  # 干净语音
    noisy_wav = 'assert/D4_750_cafe.wav'  # 含噪语音
    denoised_wav = 'assert/denoised_wav_D4_750_cafe.wav'  # 整段推理降噪后语音
    streaming_wav = 'assert/denoised_streaming_D4_750_cafe.wav'  # 流式推理降噪后语音

    if all(os.path.exists(f) for f in [clean_wav, noisy_wav, denoised_wav]):
        plot_audio_comparison(clean_wav, noisy_wav, denoised_wav, streaming_wav)
    else:
        print("❌ 必要文件不存在！请先运行降噪模型生成降噪后语音。")
        print("建议先执行：")
        print("    from your_script import predict_one_wav")
        print("    predict_one_wav('D4_750_cafe.wav')")
        print("    # 若已实现流式推理，请生成 denoised_streaming_D4_750_cafe.wav")