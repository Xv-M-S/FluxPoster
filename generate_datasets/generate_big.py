from PIL import Image, ImageDraw, ImageFont
import os

def find_suitable_font_size(text, image_size, font_path=None):
    """
    使用二分查找法寻找合适的字体大小。
    
    :param text: 要显示的文本。
    :param image_size: 图像的宽度和高度，期望文本适应于这个尺寸。
    :param font_path: 字体文件路径，默认为None，如果为None则使用默认字体。
    :return: 合适的字体大小。
    """
    lower_bound = 10  # 最小字体大小
    upper_bound = 700  # 最大字体大小
    target_size = image_size[0]  # 假设宽高相同，目标是让字体大小适应此尺寸
    
    while lower_bound <= upper_bound:
        mid_font_size = (lower_bound + upper_bound) // 2
        try:
            if font_path:
                font = ImageFont.truetype(font_path, mid_font_size)
            else:
                font = ImageFont.load_default().font_variant(size=mid_font_size)

            # 创建虚拟绘制对象以计算文本大小
            dummy_image = Image.new('RGB', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_image)
            text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            # 检查当前字体大小是否接近目标尺寸
            if abs(text_width - target_size) < 5:  # 允许有微小误差
                return mid_font_size
            elif text_width < target_size:
                lower_bound = mid_font_size + 1
            else:
                upper_bound = mid_font_size - 1
        except IOError:
            print("加载字体失败，尝试使用默认字体.")
            font_path = None  # 如果指定字体不可用，则回退到默认字体
    
    # 如果无法精确匹配，返回最接近的结果
    return (lower_bound + upper_bound) // 2


# 示例使用
output_dir = "pure_text_datasets"
count = 0
with open("/home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/ocr_weights/ppocr_keys_v1.txt", "r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    for text in lines:
        image_size = (512, 512)
        font_size = find_suitable_font_size(text=text, image_size=image_size, font_path="baige.ttf")
        print(f"字体大小：{font_size}")

        # 根据找到的字体大小创建图像
        image = Image.new('RGB', image_size, "white")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("baige.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), text, font=font)
        position = ((image_size[0]-text_bbox[2])//2, (image_size[1]-text_bbox[3])//2)
        draw.text((-50,-50), text, font=font, fill="black")

        image.save(os.path.join(output_dir, f"output_{count}.png"))
        count += 1