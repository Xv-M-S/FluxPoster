import torch
from PIL import Image
# from modelscope import AutoModel, CLIPImageProcessor
from transformers import AutoModel, CLIPImageProcessor

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-6B-448px-V1-5',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image = Image.open('/home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/posterDataSets/mask/1.png').convert('RGB')

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
print(outputs.last_hidden_state.shape) # [1,1025,3200]
print(outputs.pooler_output.shape) # [1, 3200]