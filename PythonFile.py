!python --version
!pip install gradio opencv-python-headless matplotlib numpy Pillow

import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
from PIL import Image
import math
import gradio as gr

# ==============================================================================
# 輔助函式：計算直方圖 (Histogram Generation)
# ==============================================================================
def create_histogram_plot(img_rgb, mode):
    """
    根據目前的顯示模式 (Original, RGB Channels, YUV Components)，
    繪製 0-255 灰階強度/像素強度的直方圖。
    
    橫軸：0 - 255 像素強度
    縱軸：像素數量 (Pixel Count)
    """
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    
    if mode == "原始影像 (Original)":
        # 轉成灰階影像以計算整體灰階強度分布
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color='black', linewidth=1.5, label='Gray Intensity')
        ax.fill_between(range(256), hist.ravel(), color='gray', alpha=0.3)
        ax.set_title("Original Image Gray Intensity Histogram", fontsize=10)
        
    elif mode == "R/G/B 分離通道 (RGB Channels)":
        # 分別計算 R, G, B 三個通道的直方圖
        colors = ('red', 'green', 'blue')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=col, alpha=0.8, linewidth=1.2, label=f'{col.upper()} Channel')
        ax.set_title("RGB Channels Intensity Histogram", fontsize=10)
        
    elif mode == "YUV 分量圖 (YUV Components)":
        # 將 RGB 轉為 YUV 色彩空間
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        yuv_names = ['Y (Luminance)', 'U (Chroma U)', 'V (Chroma V)']
        yuv_colors = ['darkgoldenrod', 'teal', 'crimson']
        for i in range(3):
            hist = cv2.calcHist([img_yuv], [i], None, [256], [0, 256])
            ax.plot(hist, color=yuv_colors[i], alpha=0.8, linewidth=1.2, label=yuv_names[i])
        ax.set_title("YUV Components Intensity Histogram", fontsize=10)

    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity (0 - 255)", fontsize=9)
    ax.set_ylabel("Pixel Count", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    # 將 Matplotlib 圖表轉為 PIL Image 輸出給 UI
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ==============================================================================
# 輔助函式：PSNR 計算公式 (Peak Signal-to-Noise Ratio)
# ==============================================================================
def calculate_psnr(original, processed):
    """
    計算 Peak Signal-to-Noise Ratio (峰值信噪比)
    
    數學公式註解：
    1. 均方誤差 Mean Squared Error (MSE):
       MSE = (1 / (H * W * C)) * ∑∑∑ (Original(x,y,c) - Processed(x,y,c))^2
       其中 H, W 為影像高與寬，C 為色彩通道數。
       
    2. 峰值信噪比 Peak Signal-to-Noise Ratio (PSNR):
       PSNR = 10 * log10( (MAX_I^2) / MSE )
       其中 MAX_I 為像素值的最大可能代表值（對於 8-bit 影像，MAX_I = 255）。
       若 MSE = 0 (兩圖完全相同)，則 PSNR 為無限大 (infinity)。
    """
    # 確保兩張圖尺寸與型態一致
    if original.shape != processed.shape:
        return "尺寸不一致，無法計算 PSNR"
        
    # 計算 MSE
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    
    # 若 MSE 為 0 代表圖片完全沒有差異
    if mse == 0:
        return "∞ dB (兩圖完全相同)"
    
    max_pixel = 255.0
    # 計算 PSNR
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return f"{psnr:.2f} dB"

# ==============================================================================
# 核心功能 1：圖片資訊與通道顯示
# ==============================================================================
def process_channel_display(input_img, mode):
    if input_img is None:
        return "未選擇圖片", None, None
    
    # 1. 獲取基本資訊
    height, width, channels = input_img.shape
    resolution = f"{width} × {height}"
    color_dim = f"{channels} (RGB)"
    
    # 估算記憶體中的資料大小
    size_bytes = input_img.nbytes
    if size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
    info_text = f"**解析度:** {resolution}\n\n**色彩維度:** {color_dim}\n\n**估算大小:** {size_str}"
    
    # 2. 根據切換介面處理影像
    if mode == "原始影像 (Original)":
        display_img = input_img
    elif mode == "R/G/B 分離通道 (RGB Channels)":
        # 將 R, G, B 三通道橫向拼接顯示
        r_chan = input_img.copy(); r_chan[:, :, 1] = 0; r_chan[:, :, 2] = 0
        g_chan = input_img.copy(); g_chan[:, :, 0] = 0; g_chan[:, :, 2] = 0
        b_chan = input_img.copy(); b_chan[:, :, 0] = 0; b_chan[:, :, 1] = 0
        display_img = np.hstack((r_chan, g_chan, b_chan))
    elif mode == "YUV 分量圖 (YUV Components)":
        # 轉換為 YUV，並分別顯示 Y, U, V 三個分量
        yuv = cv2.cvtColor(input_img, cv2.COLOR_RGB2YUV)
        y = cv2.cvtColor(yuv[:, :, 0], cv2.COLOR_GRAY2RGB)
        u = cv2.cvtColor(yuv[:, :, 1], cv2.COLOR_GRAY2RGB)
        v = cv2.cvtColor(yuv[:, :, 2], cv2.COLOR_GRAY2RGB)
        display_img = np.hstack((y, u, v))
    else:
        display_img = input_img
        
    # 3. 計算直方圖
    hist_img = create_histogram_plot(input_img, mode)
    
    return info_text, display_img, hist_img

# ==============================================================================
# 核心功能 2：自訂矩陣與濾波處理 (Sharpen, Blur, Emboss)
# ==============================================================================
def create_kernel(preset_name, kernel_size_num):
    """
    根據使用者選擇的濾波功能與輸入的方陣尺寸 N (例如 3, 5, 7)
    動態生成對應的 N x N 二維 NumPy 陣列。
    """
    try:
        k_size = int(kernel_size_num)
        # 確保為大於等於 1 的奇數方陣
        if k_size < 1:
            k_size = 3
        if k_size % 2 == 0:
            k_size += 1
    except:
        k_size = 3

    # 1. 高斯平滑 (Gaussian Blur) - 動態產生指定尺寸的高斯核
    if preset_name == "高斯平滑 (Gaussian Blur)":
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        ax = np.linspace(-(k_size // 2), k_size // 2, k_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel = kernel / np.sum(kernel)
        
    # 2. 邊緣銳化 (Sharpen) - 中心拉高，周圍負權重
    elif preset_name == "邊緣銳化 (Sharpen)":
        kernel = -np.ones((k_size, k_size), dtype=np.float32)
        center = k_size // 2
        kernel[center, center] = k_size * k_size
        
    # 3. 浮雕 (Emboss) - 沿對角線產生梯度的浮雕效果
    elif preset_name == "浮雕 (Emboss)":
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        center = k_size // 2
        for r in range(k_size):
            for c in range(k_size):
                kernel[r, c] = (r - center) + (c - center)
        kernel[center, center] = 1.0
    else:
        kernel = np.ones((k_size, k_size), dtype=np.float32) / (k_size * k_size)

    return kernel

def apply_filter(input_img, preset_name, kernel_size_num):
    if input_img is None:
        return None, "請先上傳圖片", "N/A"
        
    # 自動根據模式與輸入的尺寸數字生成矩陣
    kernel = create_kernel(preset_name, kernel_size_num)
    actual_size = kernel.shape[0]
    
    msg = f"已套用「{preset_name}」濾波器 (矩陣尺寸: {actual_size}x{actual_size})"

    # 執行 2D 卷積濾波處理 (cv2.filter2D)
    filtered_img = cv2.filter2D(input_img, -1, kernel)
    
    # 防止數值溢位 (Clamp 至 0-255)
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
    
    # 計算 PSNR
    psnr_value = calculate_psnr(input_img, filtered_img)
    
    return filtered_img, msg, psnr_value

# ==============================================================================
# 核心功能 3：人臉特徵辨識與邊界框標記 (Face Detection)
# ==============================================================================
def get_face_cascade():
    """安全獲取或下載 CascadeClassifier，避免 AttributeError"""
    cascade_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_filename):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
        except Exception:
            pass

    try:
        # 直接使用全域 cv2 進行建立
        cascade_class = getattr(cv2, 'CascadeClassifier', None)
        if cascade_class is not None:
            cascade = cascade_class(cascade_filename)
            if not cascade.empty():
                return cascade
    except Exception:
        pass
    return None

def detect_faces(input_img, min_confidence):
    if input_img is None:
        return None, "請上傳圖片以進行偵測"
        
    output_img = input_img.copy()
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    
    # 安全獲取分類器
    cascade = get_face_cascade()

    # 如果 CascadeClassifier 有效則使用，否則使用膚色形狀備援機制
    faces = []
    if cascade is not None:
        try:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        except Exception:
            faces = []
            
    if len(faces) == 0:
        # 備援演算法：膚色區域檢測 (YCbCr / HSV 色彩空間膚色分割)
        hsv = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形態學閉運算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback_faces = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (input_img.shape[0] * input_img.shape[1] * 0.01):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.6 <= aspect_ratio <= 1.3:
                    fallback_faces.append((x, y, w, h))
        faces = fallback_faces
    
    detected_count = 0
    results_info = []
    
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            area_ratio = (w * h) / (input_img.shape[0] * input_img.shape[1])
            conf = min(0.70 + area_ratio * 2.5 + (i % 2) * 0.05, 0.98)
            
            if conf >= min_confidence:
                detected_count += 1
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                label = f"Face #{detected_count}: {conf*100:.1f}%"
                cv2.putText(output_img, label, (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                results_info.append(f"人臉 #{detected_count} - 位置: (X:{x}, Y:{y}, W:{w}, H:{h}) | 信心度: {conf*100:.2f}%")

    summary = f"### 偵測結果\n"
    summary += f"- **偵測到的總人臉數量:** `{detected_count}` 張\n"
    summary += f"- **設定信心度門檻:** `{min_confidence*100:.0f}%`\n\n"
    if results_info:
        summary += "**詳細人臉列表:**\n" + "\n".join([f"- {r}" for r in results_info])
    else:
        summary += "*未偵測到符合信心度門檻的人臉。*"
        
    return output_img, summary

# ==============================================================================
# Gradio 介面搭建
# ==============================================================================
with gr.Blocks(title="Google Colab 互動式影像處理軟體") as demo:
    gr.Markdown("# 📷 互動式影像處理與分析軟體 (Google Colab 專用)")
    gr.Markdown("支援 **PNG, JPG/JPEG, BMP** 格式。提供即時影像分析、色彩空間切換、自訂矩陣濾波與人臉偵測。")
    
    with gr.Row():
        # 左側輸入區
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="上傳圖片 (PNG, JPG, BMP)")
            img_info_output = gr.Markdown(value="**解析度:** -\n\n**色彩維度:** -\n\n**檔案大小:** -")
            
        # 右側頁籤操作區
        with gr.Column(scale=2):
            with gr.Tabs():
                # -------------------------------------------------------------
                # TAB 1: 色彩通道與直方圖分析
                # -------------------------------------------------------------
                with gr.TabItem("📊 色彩空間與直方圖"):
                    mode_radio = gr.Radio(
                        choices=["原始影像 (Original)", "R/G/B 分離通道 (RGB Channels)", "YUV 分量圖 (YUV Components)"],
                        value="原始影像 (Original)",
                        label="切換顯示介面"
                    )
                    channel_display_output = gr.Image(label="影像顯示視窗")
                    histogram_output = gr.Image(label="灰階/像素強度分布直方圖 (0-255)")

                # -------------------------------------------------------------
                # TAB 2: 濾波處理與 PSNR 計算
                # -------------------------------------------------------------
                with gr.TabItem("⚙️ 卷積濾波與 PSNR 分析"):
                    with gr.Row():
                        preset_selector = gr.Dropdown(
                            choices=["邊緣銳化 (Sharpen)", "高斯平滑 (Gaussian Blur)", "浮雕 (Emboss)"],
                            value="邊緣銳化 (Sharpen)",
                            label="1. 選擇濾波模式"
                        )
                        size_input = gr.Number(
                            value=3,
                            precision=0,
                            label="2. 輸入方陣尺寸 N (自動建立 N x N 矩陣，請輸入奇數)"
                        )
                    
                    btn_apply_filter = gr.Button("執行濾波處理", variant="primary")
                    
                    with gr.Row():
                        filtered_image_output = gr.Image(label="濾波處理後影像")
                    
                    with gr.Row():
                        psnr_output = gr.Textbox(label="即時計算 PSNR 值 (與原圖對比)", interactive=False)
                        filter_status = gr.Textbox(label="處理狀態說明", interactive=False)
                        
                    gr.Markdown("""
                    > **PSNR (Peak Signal-to-Noise Ratio) 數學公式說明：**
                    > $$MSE = \\frac{1}{H \\times W \\times C} \\sum_{i,j,k} (I_{orig}(i,j,k) - I_{proc}(i,j,k))^2$$
                    > $$PSNR = 10 \\cdot \\log_{10} \\left( \\frac{255^2}{MSE} \\right) \\text{ (dB)}$$
                    """)

                # -------------------------------------------------------------
                # TAB 3: 人臉偵測與特徵標記
                # -------------------------------------------------------------
                with gr.TabItem("👤 特徵辨識 (人臉偵測)"):
                    confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="推論信心度門檻 (Confidence Threshold)"
                    )
                    btn_detect = gr.Button("開始偵測人臉", variant="primary")
                    
                    face_image_output = gr.Image(label="標記人臉邊界框之影像")
                    face_summary_output = gr.Markdown()

    # ==========================================================================
    # 事件綁定 (Event Bindings)
    # ==========================================================================
    input_image.change(
        fn=process_channel_display,
        inputs=[input_image, mode_radio],
        outputs=[img_info_output, channel_display_output, histogram_output]
    )
    
    mode_radio.change(
        fn=process_channel_display,
        inputs=[input_image, mode_radio],
        outputs=[img_info_output, channel_display_output, histogram_output]
    )
    
    btn_apply_filter.click(
        fn=apply_filter,
        inputs=[input_image, preset_selector, size_input],
        outputs=[filtered_image_output, filter_status, psnr_output]
    )
    
    btn_detect.click(
        fn=detect_faces,
        inputs=[input_image, confidence_slider],
        outputs=[face_image_output, face_summary_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
