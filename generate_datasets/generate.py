from PIL import Image, ImageDraw, ImageFont
import random
import string

def generate_random_text_image(output_path="random_text.png"):
    # 创建512x512的白色背景图片
    img = Image.new('RGB', (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 可用字体列表（根据你的系统调整）
    try:
        fonts = [
            ImageFont.truetype("baige.ttf", random.randint(20, 60)),
            # ImageFont.truetype("arialbd.ttf", random.randint(20, 60)),
            # ImageFont.truetype("times.ttf", random.randint(20, 60)),
            # ImageFont.truetype("cour.ttf", random.randint(20, 60)),
        ]
    except:
        # 如果找不到字体，使用默认字体
        fonts = [ImageFont.load_default()]
    
    # 生成随机文字并绘制
    for _ in range(random.randint(5, 20)):  # 随机5-20个文字元素
        # 随机位置
        x = random.randint(0, 400)
        y = random.randint(0, 400)
        
        # 随机文字内容
        text_length = random.randint(1, 10)
        text = ''.join(random.choice(string.ascii_letters + string.digits + " ,.!?") 
                       for _ in range(text_length))
        
        # 随机颜色
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # 随机旋转角度
        angle = random.randint(-30, 30)
        
        # 随机选择字体
        font = random.choice(fonts)
        
        # 创建文本图像并旋转
        text_img = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((x, y), text, fill=color, font=font)
        text_img = text_img.rotate(angle, expand=1)
        
        # 合并到主图像
        img.paste(text_img, (0, 0), text_img)
    
    # 保存图片
    img.save(output_path)
    print(f"图片已生成: {output_path}")
    return img

# 使用示例
if __name__ == "__main__":
    for i in range(100):
        generate_random_text_image(f"./datasets/random_text_{i}.png")